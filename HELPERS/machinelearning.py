"""
Train PPO on RacingEnv: parallel envs, random car rows from car_data.csv + domain randomization.

Run from project root:
  python -m HELPERS.machinelearning --n-envs 8 --timesteps 200000

How the car learns to navigate: PPO maximizes discounted return from rewards (speed on-track,
penalties off-track / time, crash reset). Ray observations give local geometry; the policy does
not see the full map—it learns reactive driving that generalizes when rewards align with lap
progress. Add checkpoints / progress rewards later for tighter lap-time optimization.
"""
from __future__ import annotations

import argparse
import os
import sys

# Project root on sys.path (allows `python HELPERS/machinelearning.py` from repo root)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from HELPERS.racing_env import RacingEnv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for Machine-Learning-Racer")
    p.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Parallel actors (vectorized envs). 200 is very CPU-heavy; start small.",
    )
    p.add_argument("--timesteps", type=int, default=500_000, help="Total environment steps")
    p.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(_ROOT, "models", "ppo_racer"),
        help="Path prefix for saving (SB3 adds .zip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", type=str, default=None, help="Optional trackname from track_data.csv")
    p.add_argument(
        "--dr",
        type=float,
        default=0.05,
        help="Domain randomization +/- fraction on car stats (default 5%%)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    env_kwargs = {
        "base_dir": _ROOT,
        "domain_randomization": args.dr,
    }
    if args.track:
        env_kwargs["track_name"] = args.track

    vec_env = make_vec_env(
        RacingEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.save_path)
    vec_env.close()


if __name__ == "__main__":
    main()
