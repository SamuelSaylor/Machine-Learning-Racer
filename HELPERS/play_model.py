"""
Watch a trained PPO drive one car on the real track (pygame window).

Uses the same RacingEnv observations/actions as training. Run in a separate terminal from training.

  python -m HELPERS.play_model --model models/ppo_racer_v1.zip --track Budapest
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO

from HELPERS.racing_env import RacingEnv


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a saved PPO policy")
    p.add_argument("--model", type=str, required=True, help="Path to .zip from training (no .zip suffix is ok)")
    p.add_argument("--track", type=str, default=None, help="trackname from track_data.csv (default: random)")
    p.add_argument("--dr", type=float, default=0.05, help="Domain randomization (match training)")
    args = p.parse_args()

    path = args.model
    if not path.endswith(".zip"):
        path = path + ".zip"

    env = RacingEnv(
        base_dir=_ROOT,
        track_name=args.track,
        domain_randomization=args.dr,
        headless=False,
    )
    model = PPO.load(path, env=env)

    clock = env._pg.time.Clock()
    obs, _ = env.reset()
    running = True

    while running:
        for event in env._pg.event.get():
            if event.type == env._pg.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        env.render()
        clock.tick(60)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
