# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Train the TRANSFORMER_OPT policy and save its weights.

The per-variable-token policy is trained by behaviour cloning of a compact
NumPy MMA teacher on synthetic constrained tasks (toy-SIMP, quadratic+linear,
curved valleys; see :mod:`scp_uno.transformer_opt_core` and
:mod:`scp_uno.mma_teacher`), optionally mixed with recorded GEMSEO-MMA
trajectories of the real GGP problem family
(``scripts/collect_ggp_mma_trajectories.py``). Pure JAX on CPU.

Usage (inside the ``ggp`` conda environment)::

    python scripts/train_transformer_opt.py --steps 6000 \
        --ggp-data scp_uno/data/ggp_mma_trajectories.npz \
        --out scp_uno/weights/transformer_opt_default.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scp_uno.transformer_opt_core import (
    PolicyConfig,
    replay_trajectory_states,
    save_params,
    train_policy,
)

REPO = Path(__file__).resolve().parents[1]


def load_ggp_pools(path: Path, cfg: PolicyConfig):
    """Convert a recorded-trajectory .npz into training state pools."""
    data = np.load(path)
    move = float(data["move"]) if "move" in data else 0.01
    pools = []
    run = 0
    while f"run{run}_X" in data:
        X = np.asarray(data[f"run{run}_X"], float)
        if len(X) >= 3:
            toks, tgts = replay_trajectory_states(
                X,
                np.asarray(data[f"run{run}_G0"], float),
                np.asarray(data[f"run{run}_C"], float),
                np.asarray(data[f"run{run}_G1"], float),
                move,
                cfg,
            )
            pools.append((toks, tgts))
            print(f"  {path.name} run{run}: {len(toks)} states (d={X.shape[1]})")
        run += 1
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-states", type=int, default=48)
    parser.add_argument("--ggp-data", type=Path, nargs="*", default=[],
                        help="Recorded GGP MMA trajectory .npz files to mix in.")
    parser.add_argument("--ggp-prob", type=float, default=0.3,
                        help="Probability of drawing a batch from --ggp-data.")
    parser.add_argument("--out", type=Path,
                        default=REPO / "scp_uno" / "weights" / "transformer_opt_default.npz")
    args = parser.parse_args()

    cfg = PolicyConfig()
    pools = []
    for path in args.ggp_data:
        pools.extend(load_ggp_pools(path, cfg))

    t0 = time.time()
    losses = []
    params = train_policy(
        cfg, steps=args.steps, lr=args.lr, batch_states=args.batch_states,
        seed=args.seed, extra_data=pools or None, extra_prob=args.ggp_prob,
        on_step=lambda i, l: losses.append(l),
    )
    print(f"trained {args.steps} steps in {time.time() - t0:.0f}s: "
          f"loss {losses[0]:.4f} -> {sum(losses[-50:]) / min(50, len(losses)):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_params(args.out, params, cfg)
    print(f"saved {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
