"""
Train PPO on RacingEnv: parallel envs, random car rows from car_data.csv + domain randomization.

Run from project root:
  python -m HELPERS.machinelearning --n-envs 8 --timesteps 200000
  python -m HELPERS.machinelearning --tensorboard ./tb_logs --timesteps 500000
  python -m HELPERS.machinelearning --render --track Budapest --timesteps 50000
  python -m HELPERS.machinelearning --render --render-grid 2 --n-envs 4 --track Budapest --timesteps 50000

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
from typing import Any, Optional

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


def _unwrap_racing_env(e: Any) -> Optional[RacingEnv]:
    cur: Any = e
    for _ in range(24):
        if isinstance(cur, RacingEnv):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "unwrapped"):
            cur = cur.unwrapped
        else:
            break
    return None


class _PygameTrainRenderCallback(BaseCallback):
    """
    Composite one or more RacingEnv instances into a single pygame window (all envs stay headless;
    frames are built offscreen and scaled into a grid).
    """

    def __init__(self, grid_side: int = 1, window_scale: float = 0.85) -> None:
        super().__init__()
        self.grid_side = max(1, int(grid_side))
        self.window_scale = max(0.25, min(1.0, float(window_scale)))
        self._grid_dims: tuple[int, int] = (1, 1)
        self._cell_px: tuple[int, int] = (800, 800)
        self._ready = False

    def _ensure_window(self) -> bool:
        if self._ready:
            return True
        venv = self.training_env
        if not hasattr(venv, "envs") or len(venv.envs) == 0:
            return False
        re0 = _unwrap_racing_env(venv.envs[0])
        if re0 is None:
            return False
        pg = re0._pg
        gs = self.grid_side
        if gs <= 1:
            cw = max(320, int(1000 * self.window_scale))
            ch = cw
            win_w, win_h = cw, ch
            self._grid_dims = (1, 1)
        else:
            cw = ch = max(240, 1000 // gs)
            win_w = cw * gs
            win_h = ch * gs
            self._grid_dims = (gs, gs)
        self._cell_px = (cw, ch)
        os.environ.pop("SDL_VIDEODRIVER", None)
        pg.display.quit()
        pg.display.init()
        pg.display.set_mode((win_w, win_h))
        pg.display.set_caption("PPO training")
        self._ready = True
        return True

    def _on_training_start(self) -> None:
        self._ensure_window()

    def _on_step(self) -> bool:
        if not self._ensure_window():
            return True
        venv = self.training_env
        re0 = _unwrap_racing_env(venv.envs[0])
        if re0 is None:
            return True
        pg = re0._pg
        screen = pg.display.get_surface()
        if screen is None:
            return True
        cols, rows = self._grid_dims
        n_show = min(len(venv.envs), cols * rows)
        screen.fill((0, 0, 0))
        cw, ch = self._cell_px
        for idx in range(n_show):
            re = _unwrap_racing_env(venv.envs[idx])
            if re is None:
                continue
            surf = re.render_frame_surface()
            if surf is None:
                continue
            scaled = pg.transform.smoothscale(surf, (cw, ch))
            col = idx % cols
            row = idx // cols
            screen.blit(scaled, (col * cw, row * ch))
        pg.display.flip()
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
        help="Open a pygame window while training (DummyVecEnv, slower). Uses headless sim + composited frames.",
    )
    p.add_argument(
        "--render-grid",
        type=int,
        default=1,
        metavar="N",
        help="Show an N×N grid of parallel envs (e.g. 2 => four cars). n_envs is raised to at least N×N.",
    )
    p.add_argument(
        "--render-window-scale",
        type=float,
        default=0.85,
        help="With --render and --render-grid 1, scale the single view (0.25–1.0 of 1000 px).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    n_envs = args.n_envs
    rg = max(1, int(args.render_grid))
    if args.render and rg > 1:
        need = rg * rg
        if n_envs < need:
            print(
                f"NOTE: --render-grid {rg} needs at least {need} envs; raising n_envs from {n_envs} to {need}.",
                file=sys.stderr,
            )
            n_envs = need

    env_kwargs = {
        "base_dir": _ROOT,
        "domain_randomization": args.dr,
        # Training render composites offscreen frames; keep all envs headless for one shared window.
        "headless": True,
    }
    if args.render:
        env_kwargs["window_scale"] = float(args.render_window_scale)
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
        callbacks.append(
            _PygameTrainRenderCallback(
                grid_side=rg,
                window_scale=float(args.render_window_scale),
            )
        )
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
