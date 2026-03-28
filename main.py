##################################################################################
ENABLE_GAME = True
# Set to FALSE if you only wish to watch the model training LIVE!
##################################################################################
# Blue (NPC) car: ML_RACER_MODEL=models/your_policy.zip → PPO; unset = arrow keys.
# Stochastic policy (default) matches training and usually moves; ML_RACER_DETERMINISTIC=1 = argmax
# and often picks "coast" at rest → car stands still (same as HELPERS/play_model.py).
##################################################################################

import os
import sys
import time

import numpy as np
import pandas as pd
import pygame

from HELPERS.racecar import RaceCar

AGENT_MODEL_PATH = os.environ.get("ML_RACER_MODEL", None)
# Default False (sample actions like training). Set ML_RACER_DETERMINISTIC=1 only if you want argmax.
AGENT_DETERMINISTIC = os.environ.get("ML_RACER_DETERMINISTIC", "").lower() in ("1", "true", "yes")

pygame.init()
## audio stuff 
engine_playing = False
audio = pygame.mixer.Sound("ASSETS/AUDIO/f1_sound.mp3")

# --- Get the track information --- #
df = pd.read_csv('ASSETS/DATA/track_data.csv')
track_info = df.loc[df['trackname'] == 'Budapest']
SPAWN_X = float(track_info["startingcoordx"].iloc[0])
SPAWN_Y = float(track_info["startingcoordy"].iloc[0])
SPAWN_ANG = float(track_info["angle"].iloc[0])

# --- Set up Screen --- #
# Game logic stays 1000×1000; window is scaled down on short displays (Linux panels / title bar).
WIDTH, HEIGHT = 1000, 1000
_info = pygame.display.Info()
_cw = _info.current_w or 1920
_ch = _info.current_h or 1080
_margin_x, _margin_y = 48, 120
_max_w = max(400, _cw - _margin_x)
_max_h = max(400, _ch - _margin_y)
# Up to ~30% larger than the old 1000×1000 cap when the display has room (was min(1.0, ...)).
_scale = min(1.3, _max_w / WIDTH, _max_h / HEIGHT)
WIN_W = max(320, int(round(WIDTH * _scale)))
WIN_H = max(320, int(round(HEIGHT * _scale)))
screen = pygame.display.set_mode((WIN_W, WIN_H))
canvas = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption(track_info['trackname'].iloc[0])
clock = pygame.time.Clock()

# --- Load Images --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NPC car image
npc_car_img = pygame.image.load(os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'BlueRacer.png')).convert_alpha()
npc_car_img = pygame.transform.scale(npc_car_img, (10, 20))

# Player car image
player_car_img = pygame.image.load(os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'RedRacer.png')).convert_alpha()
player_car_img = pygame.transform.scale(player_car_img, (10, 20))

fin = pygame.image.load(os.path.join(BASE_DIR, 'ASSETS', 'FINISHEDMESSAGE.png')).convert_alpha()
fin = pygame.transform.scale(fin, (1000, 1000))

# Track / Deadzone / Background
track_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', track_info['dirname'].iloc[0])
track_img = pygame.image.load(os.path.join(track_path, "BOUNDARY.png")).convert_alpha()
track_img = pygame.transform.scale(track_img, (1000, 1000))
deadzone_img = pygame.image.load(os.path.join(track_path, "DEADZONE.png")).convert_alpha()
deadzone_img = pygame.transform.scale(deadzone_img, (1000, 1000))
background_path = os.path.join(track_path, 'COSMETIC.png')
background_img = pygame.image.load(background_path).convert()
background_img = pygame.transform.scale(background_img, (1000, 1000))

# --- Create Masks --- #
track_mask = pygame.mask.from_surface(track_img)
boundary_mask = pygame.mask.from_surface(deadzone_img)

# --- Create Cars --- #
_track_name = str(track_info["trackname"].iloc[0])

agent_env = None
agent_model = None
npc_car = None


def _resolve_sb3_zip_path(raw: str, base_dir: str) -> str:
    p = os.path.expanduser(raw.strip())
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    if os.path.isfile(p):
        return p
    if not p.lower().endswith(".zip"):
        z = p + ".zip"
        if os.path.isfile(z):
            return z
    raise FileNotFoundError(
        f"No model at {p!r}. Set ML_RACER_MODEL to a trained PPO .zip (e.g. models/ppo_racer_v1.zip)."
    )


def _n_rays_from_sb3_model(model) -> int:
    """RacingEnv: obs_dim = 7 + n_rays."""
    sp = model.observation_space
    shape = getattr(sp, "shape", None)
    if shape is None or len(shape) != 1:
        raise ValueError("Expected a 1-D Box observation from the checkpoint.")
    n_rays = int(shape[0]) - 7
    if n_rays < 1:
        raise ValueError(f"Bad observation dim {shape[0]}; expected 7 + n_rays.")
    return n_rays


if AGENT_MODEL_PATH:
    from stable_baselines3 import PPO

    from HELPERS.racing_env import RacingEnv

    _mp = _resolve_sb3_zip_path(str(AGENT_MODEL_PATH), BASE_DIR)
    agent_model = PPO.load(_mp, device="cpu", print_system_info=False)
    _agent_n_rays = _n_rays_from_sb3_model(agent_model)
    agent_env = RacingEnv(
        base_dir=BASE_DIR,
        track_name=_track_name,
        n_rays=_agent_n_rays,
        headless=True,
        train_log=False,
        domain_randomization=0.0,
        embed_pygame=True,
    )
    agent_env.reset(seed=0)
    npc_car = agent_env.car
    # Match main.py spawn exactly (RacingEnv reset uses the same CSV row, but sync here so
    # PPO rays / respawn state cannot drift from the red car).
    npc_car.car_pos[0] = SPAWN_X
    npc_car.car_pos[1] = SPAWN_Y
    npc_car.angle = SPAWN_ANG
    npc_car.speed = 0.0
    agent_env._respawn_cp = (SPAWN_X, SPAWN_Y)
    agent_env._last_angle = SPAWN_ANG
    if AGENT_DETERMINISTIC:
        print(
            "Note: ML_RACER_DETERMINISTIC=1 uses policy argmax; at rest that is often "
            "'coast' (no throttle) so the car may not move. Run without it for stochastic "
            "actions (same as training): ML_RACER_MODEL=... python main.py",
            file=sys.stderr,
        )
else:
    npc_car = RaceCar(SPAWN_X, SPAWN_Y, SPAWN_ANG)

player = None
if ENABLE_GAME:
    player = RaceCar(SPAWN_X, SPAWN_Y, SPAWN_ANG)

checkpoint_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', track_info['dirname'].iloc[0], 'CHECKPOINTS')
checkpoints = [
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTZERO.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTONE.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTTWO.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTTHREE.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTFOUR.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTFIVE.png")).convert_alpha(),
    pygame.image.load(os.path.join(checkpoint_path, "CHECKPOINTSIXSEVEN.png")).convert_alpha(),
]

for i in range(len(checkpoints)-1):
    checkpoints[i] = pygame.transform.scale(checkpoints[i], (1000, 1000))

j=0
active_cpmask = pygame.mask.from_surface(checkpoints[j])
#active_CP = checkpoints[0]
respawn_CP = (SPAWN_X, SPAWN_Y)

# --- On-screen WASD (bottom-left): mouse / touch when keyboard over remote desktop is unreliable ---
_BTN = 56
_BTN_GAP = 8
_PAD = 16


def _wasd_button_rects():
    ay = HEIGHT - _PAD - _BTN
    ax = _PAD
    sx = _PAD + _BTN + _BTN_GAP
    dx = _PAD + 2 * (_BTN + _BTN_GAP)
    wy = ay - _BTN - _BTN_GAP
    wx = sx
    return {
        "w": pygame.Rect(wx, wy, _BTN, _BTN),
        "a": pygame.Rect(ax, ay, _BTN, _BTN),
        "s": pygame.Rect(sx, ay, _BTN, _BTN),
        "d": pygame.Rect(dx, ay, _BTN, _BTN),
    }


WASD_RECTS = _wasd_button_rects()
_WASD_FONT = pygame.font.Font(None, 34)


def _wasd_pointer_holds():
    """True while primary button held inside each rect (same frame semantics as get_pressed keys)."""
    if not pygame.mouse.get_pressed()[0]:
        return {"w": False, "a": False, "s": False, "d": False}
    mx, my = pygame.mouse.get_pos()
    # Mouse is in window pixels; rects are in 1000×1000 game space when using canvas scale.
    gx = mx * WIDTH / WIN_W
    gy = my * HEIGHT / WIN_H
    return {k: r.collidepoint(gx, gy) for k, r in WASD_RECTS.items()}


def draw_wasd_overlay(surf, holds):
    """Draw on the 1000×1000 canvas after gameplay so controls stay on top."""
    for key, r in WASD_RECTS.items():
        down = holds.get(key, False)
        fill = (70, 130, 210) if down else (48, 48, 56)
        pygame.draw.rect(surf, fill, r, border_radius=10)
        pygame.draw.rect(surf, (210, 215, 225), r, 2, border_radius=10)
        t = _WASD_FONT.render(key.upper(), True, (250, 250, 252))
        surf.blit(t, t.get_rect(center=r.center))


# --- Main Loop --- #
running = True
_pos_print_frame = 0
_agent_snap_to_player_done = False
# Wall clock for the snap: pygame.Clock.tick(60) returns 0 when FPS < 60, so don't use summed dt.
_game_start_mono = time.monotonic()

while running:
    dt = clock.tick(60) / 1000.0

    # Bandaid: PPO env spawn can disagree with main on some setups — after 1s snap blue to red once.
    if (
        not _agent_snap_to_player_done
        and agent_env is not None
        and player is not None
        and (time.monotonic() - _game_start_mono) >= 1.0
    ):
        _agent_snap_to_player_done = True
        px, py = float(player.car_pos[0]), float(player.car_pos[1])
        pa = float(player.angle)
        # Always write through env.car so we stay synced if anything held a stale reference.
        c = agent_env.car
        c.car_pos[0] = px
        c.car_pos[1] = py
        c.angle = pa
        c.speed = 0.0
        npc_car = c
        agent_env._respawn_cp = (px, py)
        agent_env._last_angle = pa

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    wasd_holds = _wasd_pointer_holds() if (ENABLE_GAME and player) else None

    # --- Player input (keyboard + bottom-left on-screen WASD for remote desktop) ---
    pinput_accel = 0
    pinput_dir = 0
    keys = pygame.key.get_pressed()
    if wasd_holds:
        tw, ta, ts, td = wasd_holds["w"], wasd_holds["a"], wasd_holds["s"], wasd_holds["d"]
    else:
        tw = ta = ts = td = False
    if keys[pygame.K_w] or tw:
        pinput_accel = 1
    elif keys[pygame.K_s] or ts:
        pinput_accel = -1
    if keys[pygame.K_a] or ta:
        pinput_dir = -1
    elif keys[pygame.K_d] or td:
        pinput_dir = 1

    # --- NPC input (PPO or arrow keys) ---
    input_accel = 0.0
    input_dir = 0.0
    if agent_model is not None and agent_env is not None:
        obs = np.asarray(agent_env._observation(), dtype=np.float32)
        action, _ = agent_model.predict(obs, deterministic=AGENT_DETERMINISTIC)
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        ia = int(np.clip(action[0], 0, 2))
        is_ = int(np.clip(action[1], 0, 2))
        agent_env._last_accel_idx = ia
        agent_env._last_steer_idx = is_
        input_accel = float([-1, 0, 1][ia])
        input_dir = float([-1, 0, 1][is_])
    else:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            input_accel = 1
        elif keys[pygame.K_DOWN]:
            input_accel = -1
        if keys[pygame.K_LEFT]:
            input_dir = -1
        elif keys[pygame.K_RIGHT]:
            input_dir = 1

    # --- Draw background ---
    canvas.blit(background_img, (0, 0))
    canvas.blit(checkpoints[j], (0, 0))
    #screen.blit(track_img, (0, 0))
    #screen.blit(deadzone_img, (0, 0))

    # --- NPC car update ---
    rotated_npc = pygame.transform.rotate(npc_car_img, npc_car.angle)
    npc_rect = rotated_npc.get_rect(center=(npc_car.car_pos[0], npc_car.car_pos[1]))
    npc_mask = pygame.mask.from_surface(rotated_npc)
    npc_offset = (npc_rect.left, npc_rect.top)
    
    npc_x = int(npc_car.car_pos[0])
    npc_y = int(npc_car.car_pos[1])
    
    npc_on_track = False
    if 0 <= npc_x < WIDTH and 0 <= npc_y < HEIGHT:
        npc_on_track = track_mask.get_at((npc_x, npc_y))

    if boundary_mask.overlap(npc_mask, npc_offset):
        npc_car.update(input_accel, input_dir, dt, 2)  # crash
    elif not npc_on_track:
        npc_car.update(input_accel, input_dir, dt, 1)  # off track
        #print("NPC off track!")
    else:
        npc_car.update(input_accel, input_dir, dt, 0)  # on track
        #print("NPC on track!")
    if agent_env is not None:
        _st, _, _ = agent_env._contact_state()
        if (
            _st == "good"
            and agent_env._cp_masks
            and agent_env._cp_idx < len(agent_env._cp_masks)
        ):
            _cm, _cr = agent_env._masks_for_car()
            _off = (_cr.left, _cr.top)
            if agent_env._cp_masks[agent_env._cp_idx].overlap(_cm, _off):
                agent_env._respawn_cp = (float(npc_car.car_pos[0]), float(npc_car.car_pos[1]))
                agent_env._cp_idx += 1
                if agent_env._cp_idx >= len(agent_env._cp_masks):
                    agent_env._lap_count += 1
                    agent_env._cp_idx = 0
    if j == len(checkpoints) - 1:
        canvas.blit(fin, (0, 0))
    # --- Player car update ---
    if player:
        rotated_player = pygame.transform.rotate(player_car_img, player.angle)
        player_rect = rotated_player.get_rect(center=(player.car_pos[0], player.car_pos[1]))
        player_mask = pygame.mask.from_surface(rotated_player)
        player_offset = (player_rect.left, player_rect.top)
        friction = 0 if track_mask.overlap(player_mask, player_offset) else 1
        player.update(pinput_accel, pinput_dir, dt, friction)
        canvas.blit(rotated_player, player_rect.topleft)
        is_stopped = abs(player.speed) < 0.01

        if is_stopped:
            if engine_playing:
                audio.stop()
                engine_playing = False
        else:
            if not engine_playing:
                audio.play(loops=-1)
                engine_playing = True

        if boundary_mask.overlap(player_mask, player_offset):
            player.car_pos[0],player.car_pos[1]=respawn_CP[0],respawn_CP[1] 

        if active_cpmask.overlap(player_mask,player_offset):
            j+=1
            active_cpmask=pygame.mask.from_surface(checkpoints[j])
            respawn_CP = (player.car_pos[0],player.car_pos[1])

    rotated_npc = pygame.transform.rotate(npc_car_img, npc_car.angle)
    npc_rect = rotated_npc.get_rect(center=(npc_car.car_pos[0], npc_car.car_pos[1]))
    canvas.blit(rotated_npc, npc_rect.topleft)
    if ENABLE_GAME and player and wasd_holds is not None:
        draw_wasd_overlay(canvas, wasd_holds)
    if WIN_W == WIDTH and WIN_H == HEIGHT:
        screen.blit(canvas, (0, 0))
    else:
        _scaled = pygame.transform.smoothscale(canvas, (WIN_W, WIN_H))
        screen.blit(_scaled, (0, 0))
    _pos_print_frame += 1
    if _pos_print_frame >= 30:
        _pos_print_frame = 0
        ax, ay = float(npc_car.car_pos[0]), float(npc_car.car_pos[1])
        if player is not None:
            px, py = float(player.car_pos[0]), float(player.car_pos[1])
            print(
                f"agent ({ax:.1f}, {ay:.1f})  player ({px:.1f}, {py:.1f})",
                flush=True,
            )
        else:
            print(f"agent ({ax:.1f}, {ay:.1f})  player (disabled)", flush=True)
    pygame.display.flip()

pygame.quit()
sys.exit()