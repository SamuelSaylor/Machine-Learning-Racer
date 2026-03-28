"""
Gymnasium racing environment aligned with main.py mask logic and RaceCar physics.

Learning signal: forward/on-track/checkpoint rewards; penalties for reverse and leaving the surface;
ray observations plus checkpoint progress. Checkpoints follow ASSETS/TRACKS/<track>/CHECKPOINTS/*.png
(same order as main.py).
"""
from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from HELPERS.racecar import RaceCar

# Observation vector layout (Box shape (16,) with n_rays=9, default). See RacingEnv.observation_space.
OBSERVATION_LAYOUT = """
Index  Meaning (all float32, roughly in [-1, 1] unless noted)
-----  --------
0      Last accel command: -1 (brake), 0 (coast), 1 (throttle)
1      Last steer command: -1 (left), 0 (straight), 1 (right)
2      Speed / max_speed (clamped to [-1, 1])
3..11  Ray distances: 9 rays from -60° to +60° relative to car forward, normalized by ray_max
12     +1 if last contact was "good" (on BOUNDARY, not in DEADZONE), else -1
13     +1 if "off_track", else -1
14     +1 if "boundary" (deadzone), else -1
15     Checkpoint progress: 2*cp_idx/(n_cp-1)-1 when n_cp>=2, else 0 (which checkpoint is next)

Action: MultiDiscrete([3,3]) → accel_idx, steer_idx → maps to {-1,0,1} × {-1,0,1} like main.py keys.
"""

# Lazy so training can use SDL dummy driver; play_model uses a real window (separate process).
_pg: Any = None


def _ensure_pygame(headless: bool, display_size: Tuple[int, int]) -> Any:
    global _pg
    if _pg is not None:
        return _pg
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pygame_mod

    _pg = pygame_mod
    pygame_mod.init()
    pygame_mod.mixer.quit()
    if headless:
        pygame_mod.display.set_mode((1, 1))
    else:
        pygame_mod.display.set_mode(display_size)
    return _pg


def _load_surface_rgba(pg: Any, path: str, size: Tuple[int, int]) -> Any:
    surf = pg.image.load(path).convert_alpha()
    return pg.transform.scale(surf, size)


def _build_numpy_road_mask(
    boundary_rgba: np.ndarray,
    deadzone_rgba: np.ndarray,
    deadzone_alpha_bad: float = 30.0,
    boundary_min_rgb: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate 'good' driving pixels for fast raycasts (center-pixel semantics)."""
    b = boundary_rgba[:, :, :3].astype(np.float32)
    on_surface = np.max(b, axis=2) > boundary_min_rgb
    da = deadzone_rgba[:, :, 3].astype(np.float32)
    dead = da > deadzone_alpha_bad
    good = on_surface & (~dead)
    return good.astype(np.bool_), dead.astype(np.bool_)


def _load_rgba_np(path: str, size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA").resize(size, Image.NEAREST))


_CHECKPOINT_FILENAMES = (
    "CHECKPOINTONE.png",
    "CHECKPOINTTWO.png",
    "CHECKPOINTTHREE.png",
    "CHECKPOINTFOUR.png",
    "CHECKPOINTFIVE.png",
)

# On racing line: discourage weaving — steer index changes (snappy left/right) and turning at speed.
# Not-straight is scaled by forward_n**2 so slow turns in hairpins stay cheap; fast S-shapes cost more.
_STEER_PENALTY_NOT_STRAIGHT: float = 0.008
_STEER_PENALTY_INDEX_CHANGE: float = 0.003  # × abs(steer_idx - prev_steer_idx), max delta 2


class RacingEnv(gym.Env):
    """
    One car per env. Actions match main.py: accel in {-1,0,1}, steer in {-1,0,1}.
    Observation: last accel, last steer, normalized speed, ray distances, contact flags.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_dir: str,
        track_name: Optional[str] = None,
        n_rays: int = 9,
        ray_step: float = 6.0,
        ray_max: float = 320.0,
        domain_randomization: float = 0.05,
        max_episode_steps: int = 2500,
        car_image_name: str = "BlueRacer.png",
        screen_size: Tuple[int, int] = (1000, 1000),
        seed: Optional[int] = None,
        headless: bool = True,
        window_scale: float = 0.8,
        train_log: bool = False,
        physics_substeps: int = 1,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.screen_size = screen_size
        self.n_rays = n_rays
        self.ray_step = ray_step
        self.ray_max = ray_max
        self.dr = domain_randomization
        self.max_episode_steps = max_episode_steps
        self._headless = headless
        self._train_log = bool(train_log)
        self._physics_substeps = max(1, int(physics_substeps))

        # World stays screen_size (e.g. 1000×1000). Window is scaled down so the full map fits
        # on typical displays (avoids the bottom being cut off by taskbars / max window height).
        if not headless:
            ws = max(0.25, min(1.0, float(window_scale)))
            self._window_size = (
                max(320, int(screen_size[0] * ws)),
                max(320, int(screen_size[1] * ws)),
            )
        else:
            self._window_size = (1, 1)

        self._pg = _ensure_pygame(headless, self._window_size if not headless else (1, 1))
        if not headless:
            self._pg.display.set_caption("RacingEnv (policy eval)")
        # Offscreen world buffer: used for human render and for training-time compositing (headless ok).
        self._frame_buf = self._pg.Surface(screen_size)
        self._background_img: Optional[Any] = None

        tracks_path = os.path.join(base_dir, "ASSETS", "DATA", "track_data.csv")
        cars_path = os.path.join(base_dir, "ASSETS", "DATA", "car_data.csv")
        self._tracks_df = pd.read_csv(tracks_path)
        self._cars_df = pd.read_csv(cars_path)

        if track_name is None:
            self._track_name = None
        else:
            self._track_name = track_name

        self._rng = np.random.default_rng(seed)
        self._episode_steps = 0
        self._last_accel_idx = 1
        self._last_steer_idx = 1
        self._steps_on_good = 0
        self._cp_idx = 0
        self._respawn_cp: Tuple[float, float] = (0.0, 0.0)
        self._cp_masks: list[Any] = []
        self._lap_count = 0
        self._pending_episode_print: Optional[Dict[str, Any]] = None
        self._ep_log: Dict[str, Any] = {}
        self._car_surface_base = _load_surface_rgba(
            self._pg,
            os.path.join(base_dir, "ASSETS", "CARS", car_image_name),
            (20, 32),
        )

        # Spaces: same discrete commands as main.py (up/down exclusive, left/right exclusive)
        self.action_space = spaces.MultiDiscrete([3, 3])

        obs_dim = 2 + 1 + n_rays + 3 + 1  # last actions, speed, rays, flags, checkpoint progress
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.car: Optional[RaceCar] = None
        self._track_row: Optional[pd.Series] = None
        self._track_mask: Optional[Any] = None
        self._boundary_mask: Optional[Any] = None
        self._road_good: Optional[np.ndarray] = None
        self._h: int = screen_size[1]
        self._w: int = screen_size[0]
        self._track_cache: Dict[str, Tuple[Any, Any, np.ndarray, list[Any]]] = {}

        self._load_track_assets(self._pick_track_row())

    def _load_checkpoint_masks(self, tdir: str) -> list[Any]:
        """Sequential checkpoint masks (same order as main.py). Missing folder/files → empty list."""
        cp_dir = os.path.join(tdir, "CHECKPOINTS")
        out: list[Any] = []
        if not os.path.isdir(cp_dir):
            return out
        for name in _CHECKPOINT_FILENAMES:
            path = os.path.join(cp_dir, name)
            if not os.path.isfile(path):
                continue
            surf = _load_surface_rgba(self._pg, path, self.screen_size)
            out.append(self._pg.mask.from_surface(surf))
        return out

    def _checkpoint_obs(self) -> float:
        n = len(self._cp_masks)
        if n == 0:
            return 0.0
        if n == 1:
            return -1.0 if self._cp_idx == 0 else 1.0
        return float(2.0 * self._cp_idx / float(n - 1) - 1.0)

    def _pick_track_row(self) -> pd.Series:
        if self._track_name is not None:
            row = self._tracks_df.loc[self._tracks_df["trackname"] == self._track_name]
            if row.empty:
                raise ValueError(f"Unknown trackname={self._track_name}")
            return row.iloc[0]
        idx = int(self._rng.integers(0, len(self._tracks_df)))
        return self._tracks_df.iloc[idx]

    def _pick_car_stats(self) -> Dict[str, float]:
        idx = int(self._rng.integers(0, len(self._cars_df)))
        row = self._cars_df.iloc[idx]
        lo, hi = 1.0 - self.dr, 1.0 + self.dr

        def jitter(x: float) -> float:
            return float(x) * float(self._rng.uniform(lo, hi))

        ms = jitter(float(row["max_speed"]))
        acc = jitter(float(row["acceleration"]))
        brk = float(row["braking"])
        brk = abs(brk) * float(self._rng.uniform(lo, hi))
        brk = -brk
        turn = jitter(float(row["turn_speed"]))
        return {
            "max_speed": ms,
            "acceleration": acc,
            "braking": brk,
            "turn_speed": turn,
        }

    def _load_track_assets(self, track_row: pd.Series) -> None:
        dirname = str(track_row["dirname"])
        if dirname in self._track_cache:
            tm, bm, rg, cp_masks = self._track_cache[dirname]
            self._track_mask = tm
            self._boundary_mask = bm
            self._road_good = rg
            self._cp_masks = cp_masks
            self._track_row = track_row
            tdir = os.path.join(self.base_dir, "ASSETS", "TRACKS", dirname)
            cosmetic_path = os.path.join(tdir, "COSMETIC.png")
            bg = self._pg.image.load(cosmetic_path).convert()
            self._background_img = self._pg.transform.scale(bg, self.screen_size)
            return

        tdir = os.path.join(self.base_dir, "ASSETS", "TRACKS", dirname)
        boundary_path = os.path.join(tdir, "BOUNDARY.png")
        deadzone_path = os.path.join(tdir, "DEADZONE.png")

        b_surf = _load_surface_rgba(self._pg, boundary_path, self.screen_size)
        d_surf = _load_surface_rgba(self._pg, deadzone_path, self.screen_size)
        track_m = self._pg.mask.from_surface(b_surf)
        boundary_m = self._pg.mask.from_surface(d_surf)

        b_rgba = _load_rgba_np(boundary_path, self.screen_size)
        d_rgba = _load_rgba_np(deadzone_path, self.screen_size)
        road_good, _ = _build_numpy_road_mask(b_rgba, d_rgba)

        cp_masks = self._load_checkpoint_masks(tdir)

        self._track_cache[dirname] = (track_m, boundary_m, road_good, cp_masks)
        self._track_mask = track_m
        self._boundary_mask = boundary_m
        self._road_good = road_good
        self._cp_masks = cp_masks
        self._track_row = track_row

        cosmetic_path = os.path.join(tdir, "COSMETIC.png")
        bg = self._pg.image.load(cosmetic_path).convert()
        self._background_img = self._pg.transform.scale(bg, self.screen_size)

    def _masks_for_car(self) -> Tuple[Any, Any]:
        rotated = self._pg.transform.rotate(self._car_surface_base, self.car.angle)
        rect = rotated.get_rect(center=(self.car.car_pos[0], self.car.car_pos[1]))
        m = self._pg.mask.from_surface(rotated)
        return m, rect

    def _contact_state(self) -> Tuple[str, Any, Tuple[int, int]]:
        """Return 'boundary', 'off_track', or 'good' — same order as main.py."""
        car_mask, rect = self._masks_for_car()
        offset = (rect.left, rect.top)
        if self._boundary_mask.overlap(car_mask, offset):
            return "boundary", car_mask, offset
        if not self._track_mask.overlap(car_mask, offset):
            return "off_track", car_mask, offset
        return "good", car_mask, offset

    def _forward_basis(self) -> Tuple[float, float, float, float]:
        """Unit forward (fx, fy) and right (rx, ry) in screen coords (matches RaceCar)."""
        rad = math.radians(-self.car.angle)
        fx, fy = math.sin(rad), -math.cos(rad)
        rx, ry = fy, -fx
        return fx, fy, rx, ry

    def _ray_distances(self) -> np.ndarray:
        assert self._road_good is not None
        cx, cy = float(self.car.car_pos[0]), float(self.car.car_pos[1])
        fx, fy, rx, ry = self._forward_basis()
        half_span = 60.0
        if self.n_rays <= 1:
            angles = [0.0]
        else:
            angles = [half_span * (2.0 * i / (self.n_rays - 1) - 1.0) for i in range(self.n_rays)]
        out = np.zeros(self.n_rays, dtype=np.float32)
        for i, deg in enumerate(angles):
            rad = math.radians(deg)
            dx = fx * math.cos(rad) + rx * math.sin(rad)
            dy = fy * math.cos(rad) + ry * math.sin(rad)
            dist = 0.0
            while dist < self.ray_max:
                dist += self.ray_step
                px = cx + dx * dist
                py = cy + dy * dist
                xi, yi = int(px), int(py)
                if xi < 0 or yi < 0 or xi >= self._w or yi >= self._h:
                    break
                if not self._road_good[yi, xi]:
                    break
            out[i] = min(dist, self.ray_max) / self.ray_max
        return out

    def _observation(self) -> np.ndarray:
        la = np.array([-1.0, 0.0, 1.0], dtype=np.float32)[self._last_accel_idx]
        ls = np.array([-1.0, 0.0, 1.0], dtype=np.float32)[self._last_steer_idx]
        spd = float(self.car.speed) / max(float(self.car.max_speed), 1e-6)
        spd = float(np.clip(spd, -1.0, 1.0))
        rays = self._ray_distances()
        state, _, _ = self._contact_state()
        b_good = 1.0 if state == "good" else -1.0
        b_off = 1.0 if state == "off_track" else -1.0
        b_bnd = 1.0 if state == "boundary" else -1.0
        cp = np.array([self._checkpoint_obs()], dtype=np.float32)
        return np.concatenate(
            [
                np.array([la, ls, spd], dtype=np.float32),
                rays,
                np.array([b_good, b_off, b_bnd], dtype=np.float32),
                cp,
            ],
            axis=0,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._train_log and self._pending_episode_print is not None:
            p = self._pending_episode_print
            ep_len = max(1, int(p.get("ep_len", 1)))
            mismatch_pct = 100.0 * float(p.get("steps_mapping_mismatch", 0)) / float(ep_len)
            throttle_pct = 100.0 * float(p.get("steps_throttle_cmd", 0)) / float(ep_len)
            moving_fwd_pct = 100.0 * float(p.get("steps_moving_forward", 0)) / float(ep_len)
            rs = float(p.get("reward_sum", 0.0))
            ps = float(p.get("punish_sum", 0.0))
            total_r = rs - ps
            print(
                "[RacingEnv] "
                f"ep_return={p.get('ep_return', 0.0):.2f} len={ep_len} "
                f"laps={p.get('lap_count', 0)} | "
                f"reward={rs:.2f} punish={ps:.2f} total_reward={total_r:.2f} "
                f"(total_reward = reward - punish) | "
                f"r_fwd={p.get('r_forward', 0.0):.2f} r_on_track={p.get('r_on_track', 0.0):.2f} "
                f"r_cp={p.get('r_checkpoint', 0.0):.2f} r_lap={p.get('r_lap', 0.0):.2f} "
                f"r_thr_bonus={p.get('r_throttle_bonus', 0.0):.2f} "
                f"p_rev={p.get('p_backward', 0.0):.2f} p_steer={p.get('p_steer', 0.0):.2f} "
                f"p_off={p.get('p_off_track', 0.0):.2f} p_bnd={p.get('p_boundary', 0.0):.2f} | "
                f"throttle_cmd={throttle_pct:.0f}% moving_forward={moving_fwd_pct:.0f}% "
                f"MAP_MISMATCH={mismatch_pct:.0f}% (throttle & speed < -max(3,12% max_speed); "
                f"transient reverse while accelerating ignored)",
                flush=True,
            )
            self._pending_episode_print = None

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode_steps = 0
        self._last_accel_idx = 1
        self._last_steer_idx = 1
        self._steps_on_good = 0
        self._cp_idx = 0
        self._lap_count = 0
        self._ep_log = {
            "ep_return": 0.0,
            "ep_len": 0,
            "r_forward": 0.0,
            "r_on_track": 0.0,
            "r_checkpoint": 0.0,
            "r_lap": 0.0,
            "r_throttle_bonus": 0.0,
            "p_backward": 0.0,
            "p_steer": 0.0,
            "p_off_track": 0.0,
            "p_boundary": 0.0,
            "reward_sum": 0.0,
            "punish_sum": 0.0,
            "steps_throttle_cmd": 0,
            "steps_moving_forward": 0,
            "steps_mapping_mismatch": 0,
        }

        track_row = self._pick_track_row()
        self._load_track_assets(track_row)
        stats = self._pick_car_stats()

        sx = float(track_row["startingcoordx"])
        # Older CSVs used typo "startingcordy"; current uses "startingcoordy"
        if "startingcoordy" in track_row.index:
            sy = float(track_row["startingcoordy"])
        else:
            sy = float(track_row["startingcordy"])
        ang = float(track_row["angle"])
        self.car = RaceCar(sx, sy, ang)
        self.car.max_speed = stats["max_speed"]
        self.car.acceleration = stats["acceleration"]
        self.car.braking = stats["braking"]
        self.car.turn_speed = stats["turn_speed"]

        self._respawn_cp = (float(sx), float(sy))

        return self._observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.car is not None
        self._episode_steps += 1

        prev_steer_idx = self._last_steer_idx

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        accel_idx = int(np.clip(a[0], 0, 2))
        steer_idx = int(np.clip(a[1], 0, 2))
        input_accel = float([-1, 0, 1][accel_idx])
        input_dir = float([-1, 0, 1][steer_idx])
        self._last_accel_idx = accel_idx
        self._last_steer_idx = steer_idx

        dt = 1.0 / 60.0
        for _ in range(self._physics_substeps):
            state, _, _ = self._contact_state()
            if state == "boundary":
                friction = 2.0
            elif state == "off_track":
                friction = 1.0
            else:
                friction = 0.0
            self.car.update(input_accel, input_dir, dt, friction)

        car_mask, rect = self._masks_for_car()
        offset = (rect.left, rect.top)

        state2, _, _ = self._contact_state()
        reward = 0.0
        terminated = False
        truncated = self._episode_steps >= self.max_episode_steps

        # Reward components (logged for training diagnostics)
        r_forward = 0.0
        r_on_track = 0.0
        r_checkpoint = 0.0
        r_lap = 0.0
        r_throttle_bonus = 0.0
        p_backward = 0.0
        p_steer = 0.0
        p_off_track = 0.0
        p_boundary = 0.0

        ms = max(float(self.car.max_speed), 1e-6)
        boundary_this_step = False

        # Boundary (deadzone): respawn like main.py — no episode terminal, strong penalty
        if state2 == "boundary":
            boundary_this_step = True
            p_boundary = 10.0
            reward -= p_boundary
            self.car.car_pos[0] = self._respawn_cp[0]
            self.car.car_pos[1] = self._respawn_cp[1]
            self.car.speed = 0.0
            state2, _, _ = self._contact_state()
            car_mask, rect = self._masks_for_car()
            offset = (rect.left, rect.top)

        # Must read speed *after* possible respawn — otherwise good-state rewards use stale speed.
        speed = float(self.car.speed)
        speed_abs = abs(speed)
        speed_n = speed_abs / ms
        forward_n = max(0.0, speed) / ms

        wants_forward = accel_idx == 2
        # Log "moving forward" when speed is clearly positive (fraction of max); loose enough for slow crawl
        moving_forward = speed > max(0.15, 0.015 * ms)
        # Only flag mapping issues when still clearly reversing while throttling (not 1–2 frames recovering from reverse)
        _mismatch_thr = max(3.0, 0.12 * ms)
        mapping_mismatch = wants_forward and speed < -_mismatch_thr

        if state2 == "off_track":
            p_off_track = 0.14 + 0.004 * speed_n
            reward -= p_off_track
        elif state2 == "good" and not boundary_this_step:
            # Forward motion should dominate; passive on-track was too high (0.025*2500 drowned r_fwd).
            r_forward = 0.56 * forward_n
            r_on_track = 0.007
            reward += r_forward + r_on_track
            self._steps_on_good += 1
            # Punish backward travel on the racing line (strong — reverse is costly vs checkpoints)
            if speed < 0.0:
                rev_n = min(1.0, (-speed) / ms)
                p_backward = 0.22 * rev_n
                reward -= p_backward
            # Small bonus for choosing throttle on the racing line (reduces coast/reverse-only policies)
            if accel_idx == 2:
                r_throttle_bonus = 0.01
                reward += r_throttle_bonus

            # Penalize steering away from straight (worse at high forward speed) and rapid steer flips
            steer_delta = abs(steer_idx - prev_steer_idx)
            if steer_delta > 0:
                p_steer += _STEER_PENALTY_INDEX_CHANGE * float(steer_delta)
            if steer_idx != 1:
                p_steer += _STEER_PENALTY_NOT_STRAIGHT * (forward_n * forward_n)
            reward -= p_steer

            # Checkpoints (same order as main.py: next mask must be hit)
            if self._cp_masks and self._cp_idx < len(self._cp_masks):
                if self._cp_masks[self._cp_idx].overlap(car_mask, offset):
                    r_checkpoint = 15.0
                    reward += r_checkpoint
                    self._respawn_cp = (float(self.car.car_pos[0]), float(self.car.car_pos[1]))
                    self._cp_idx += 1
                    if self._cp_idx >= len(self._cp_masks):
                        r_lap = 35.0
                        reward += r_lap
                        self._lap_count += 1
                        self._cp_idx = 0

        # Small per-step cost (encourages finishing / progress)
        step_time_cost = 0.0008
        reward -= step_time_cost

        final_reward = 0.0
        if truncated:
            final_reward = 1.5 + 0.0005 * float(self._steps_on_good)
            reward += final_reward

        # Episode accounting: reward_sum = bonuses; punish_sum = penalties (incl. time cost); total = r − p ≈ ep_return
        step_reward_pos = (
            r_forward + r_on_track + r_checkpoint + r_lap + r_throttle_bonus + (final_reward if truncated else 0.0)
        )
        step_punish = p_backward + p_steer + p_off_track + p_boundary + step_time_cost
        self._ep_log["reward_sum"] += float(step_reward_pos)
        self._ep_log["punish_sum"] += float(step_punish)

        self._ep_log["ep_return"] += float(reward)
        self._ep_log["ep_len"] += 1
        self._ep_log["r_forward"] += r_forward
        self._ep_log["r_on_track"] += r_on_track
        self._ep_log["r_checkpoint"] += r_checkpoint
        self._ep_log["r_lap"] += r_lap
        self._ep_log["r_throttle_bonus"] += r_throttle_bonus
        self._ep_log["p_backward"] += p_backward
        self._ep_log["p_steer"] += p_steer
        self._ep_log["p_off_track"] += p_off_track
        self._ep_log["p_boundary"] += p_boundary
        if wants_forward:
            self._ep_log["steps_throttle_cmd"] += 1
        if moving_forward:
            self._ep_log["steps_moving_forward"] += 1
        if mapping_mismatch:
            self._ep_log["steps_mapping_mismatch"] += 1

        if truncated:
            self._pending_episode_print = {**dict(self._ep_log), "lap_count": self._lap_count}

        info: Dict[str, Any] = {
            "contact": state2,
            "final_reward": float(final_reward),
            "reward_total": float(reward),
            "step_reward_pos": float(step_reward_pos),
            "step_punish": float(step_punish),
            "episode_reward_sum": float(self._ep_log["reward_sum"]),
            "episode_punish_sum": float(self._ep_log["punish_sum"]),
            "r_forward": float(r_forward),
            "r_on_track": float(r_on_track),
            "r_checkpoint": float(r_checkpoint),
            "r_lap": float(r_lap),
            "r_throttle_bonus": float(r_throttle_bonus),
            "p_backward": float(p_backward),
            "p_steer": float(p_steer),
            "p_off_track": float(p_off_track),
            "p_boundary": float(p_boundary),
            "accel_cmd": float(input_accel),
            "speed": float(speed),
            "wants_forward": bool(wants_forward),
            "moving_forward": bool(moving_forward),
            "mapping_mismatch": bool(mapping_mismatch),
            "checkpoint_idx": int(self._cp_idx),
            "num_checkpoints": int(len(self._cp_masks)),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render_frame_surface(self) -> Optional[Any]:
        """Compose the full map + car to the offscreen buffer; works in headless mode (no window)."""
        if self.car is None or self._background_img is None:
            return None
        self._pg.event.pump()
        self._frame_buf.blit(self._background_img, (0, 0))
        rotated = self._pg.transform.rotate(self._car_surface_base, self.car.angle)
        rect = rotated.get_rect(center=(self.car.car_pos[0], self.car.car_pos[1]))
        self._frame_buf.blit(rotated, rect.topleft)
        return self._frame_buf

    def render(self) -> Optional[Any]:
        """Draw the full world (1000×1000) then scale to the window so the entire map stays visible."""
        if self._headless:
            return None
        surf = self.render_frame_surface()
        if surf is None:
            return None
        win = self._pg.display.get_surface()
        scaled = self._pg.transform.smoothscale(surf, self._window_size)
        win.blit(scaled, (0, 0))
        self._pg.display.flip()
        return None

    def close(self) -> None:
        pass
