"""
Train PPO on RacingEnv: parallel envs, random car rows from car_data.csv + domain randomization.

Run from project root:
  python -m HELPERS.machinelearning --n-envs 8 --timesteps 200000
  python -m HELPERS.machinelearning --tensorboard ./tb_logs --timesteps 500000
  python -m HELPERS.machinelearning --render --track Budapest --timesteps 50000   # one window, slow

TensorBoard: tensorboard --logdir ./tb_logs  (actor/critic losses, entropy, episode reward)

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

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from HELPERS.racing_env import RacingEnv


class _PygameTrainRenderCallback(BaseCallback):
    """Refresh the pygame window for env 0 (DummyVecEnv + RacingEnv headless=False)."""

    def _on_step(self) -> bool:
        venv = self.training_env
        if not hasattr(venv, "envs"):
            return True
        e = venv.envs[0]
        for _ in range(12):
            if hasattr(e, "render"):
                try:
                    e.render()
                    break
                except Exception:
                    pass
            if hasattr(e, "env"):
                e = e.env
            else:
                break
        return True


def _resolve_device(name: str) -> str:
    name = name.lower().strip()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Update your NVIDIA driver and/or reinstall PyTorch for your CUDA version "
                "(see https://pytorch.org/get-started/locally/). "
                "Run: nvidia-smi   and   python -c \"import torch; print(torch.__version__, torch.version.cuda)\""
            )
        return "cuda"
    if name == "cpu":
        return "cpu"
    raise ValueError(f"Unknown --device {name!r}; use auto, cuda, or cpu")


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
    p.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rich progress bar (needs: pip install rich tqdm). Use --no-progress-bar to disable.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Where to run the policy/value networks (env sim stays on CPU). Default: auto = cuda if available.",
    )
    p.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        metavar="DIR",
        help="Log SB3 metrics (policy loss, value loss, entropy, episode reward). "
        "Then run: tensorboard --logdir DIR",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Open one pygame window and show env 0 while training. Forces n_envs=1 and DummyVecEnv (slow).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    n_envs = args.n_envs
    if args.render:
        if n_envs != 1:
            print(
                f"NOTE: --render only supports one env; using n_envs=1 (you passed {n_envs}).",
                file=sys.stderr,
            )
        n_envs = 1

    env_kwargs = {
        "base_dir": _ROOT,
        "domain_randomization": args.dr,
        "headless": not args.render,
    }
    if args.track:
        env_kwargs["track_name"] = args.track

    if args.render:
        vec_env = make_vec_env(
            RacingEnv,
            n_envs=n_envs,
            seed=args.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
        )
    else:
        vec_env = make_vec_env(
            RacingEnv,
            n_envs=n_envs,
            seed=args.seed,
            env_kwargs=env_kwargs,
        )

    device = _resolve_device(args.device)
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for PPO networks (env stepping is always CPU-bound).")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        device=device,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=args.tensorboard,
    )
    use_pb = args.progress_bar
    if use_pb:
        try:
            import rich  # noqa: F401
        except ImportError:
            use_pb = False

    callbacks = []
    if args.render:
        callbacks.append(_PygameTrainRenderCallback())
    if args.tensorboard:
        os.makedirs(args.tensorboard, exist_ok=True)
        print(f"TensorBoard: tensorboard --logdir {args.tensorboard}")

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=use_pb,
        callback=callbacks if callbacks else None,
    )
    model.save(args.save_path)
    vec_env.close()


if __name__ == "__main__":
    main()
