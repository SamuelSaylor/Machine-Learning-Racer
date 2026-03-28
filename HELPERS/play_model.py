"""
Watch a trained PPO drive one car on the real track (pygame window).

Uses the same RacingEnv observations/actions as training. Run in a separate terminal from training.

  python -m HELPERS.play_model --model models/ppo_racer_v1.zip --track Budapest
  python -m HELPERS.play_model --model models/ppo_racer_v1 --window-scale 0.65
  python -m HELPERS.play_model --model models/ppo_racer_v1 --stochastic   # if car never moves: policy may be choosing no throttle (idx 1)

Action indices map to accel/steer {-1,0,1}. First index: 0=brake/rev, 1=coast, 2=throttle. At speed 0, coast keeps you stopped; you need idx 2 to accelerate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO

from HELPERS.racing_env import OBSERVATION_LAYOUT, RacingEnv


def _decode_action(action: np.ndarray) -> tuple[float, float]:
    """Same mapping as RacingEnv.step: indices 0,1,2 -> -1,0,1."""
    a = np.asarray(action, dtype=np.int64).reshape(-1)
    ia, is_ = int(np.clip(a[0], 0, 2)), int(np.clip(a[1], 0, 2))
    accel = float([-1, 0, 1][ia])
    steer = float([-1, 0, 1][is_])
    return accel, steer


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a saved PPO policy")
    p.add_argument("--model", type=str, required=True, help="Path to .zip from training (no .zip suffix is ok)")
    p.add_argument("--track", type=str, default=None, help="trackname from track_data.csv (default: random)")
    p.add_argument("--dr", type=float, default=0.05, help="Domain randomization (match training)")
    p.add_argument(
        "--window-scale",
        type=float,
        default=0.8,
        help="Scale the full 1000×1000 map into the window (0.25–1.0). Lower = smaller window, entire map visible.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print action, speed, position every 30 frames (stderr)",
    )
    p.add_argument(
        "--show-obs",
        action="store_true",
        help="Print observation layout (same as training) and exit",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy (not argmax). Use if the car sits still with deterministic=True.",
    )
    args = p.parse_args()

    if args.show_obs:
        print(OBSERVATION_LAYOUT.strip())
        return

    path = args.model
    if not path.endswith(".zip"):
        path = path + ".zip"

    # Load weights only — do NOT pass env= here or SB3 wraps your env in DummyVecEnv/Monitor
    # and stepping/rendering no longer matches this RacingEnv instance.
    model = PPO.load(path, device="cpu", print_system_info=False)

    env = RacingEnv(
        base_dir=_ROOT,
        track_name=args.track,
        domain_randomization=args.dr,
        headless=False,
        window_scale=args.window_scale,
    )

    if not os.environ.get("DISPLAY"):
        print(
            "WARNING: DISPLAY is not set. If you are on SSH, run on the machine's desktop "
            "or use `ssh -X` / Wayland so a pygame window can open.",
            file=sys.stderr,
        )

    clock = env._pg.time.Clock()
    obs, _ = env.reset()
    env.render()
    running = True
    frame = 0

    while running:
        for event in env._pg.event.get():
            if event.type == env._pg.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=not args.stochastic)
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        acc, steer = _decode_action(action)

        obs, _reward, terminated, truncated, info = env.step(action)
        env.render()
        clock.tick(60)
        frame += 1
        if args.debug and frame % 30 == 0:
            c = env.car
            print(
                f"idx={action.tolist()} -> accel={acc} steer={steer} | "
                f"speed={getattr(c, 'speed', None)} pos={getattr(c, 'car_pos', None)} "
                f"contact={info.get('contact')}",
                file=sys.stderr,
            )

        if terminated or truncated:
            obs, _ = env.reset()
            env.render()

    env.close()


if __name__ == "__main__":
    main()
