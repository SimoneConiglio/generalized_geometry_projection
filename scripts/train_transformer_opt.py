# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Train the TRANSFORMER_OPT policy and save its weights.

The policy is trained by behaviour cloning of a privileged teacher on
synthetic differentiable tasks (quadratics, two-well, curved valleys) of
random dimension; see :mod:`scp_uno.transformer_opt_core`. Training is pure
JAX on CPU (a few minutes for the default 3000 steps).

Usage (inside the ``ggp`` conda environment)::

    python scripts/train_transformer_opt.py \
        --steps 3000 --out scp_uno/weights/transformer_opt_default.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scp_uno.transformer_opt_core import PolicyConfig, save_params, train_policy

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--batch-heads", type=int, default=4,
                        help="q: proposal heads = points per iteration.")
    parser.add_argument("--out", type=Path,
                        default=REPO / "scp_uno" / "weights" / "transformer_opt_default.npz")
    args = parser.parse_args()

    cfg = PolicyConfig(latent_dim=args.latent_dim, n_heads_out=args.batch_heads)
    t0 = time.time()
    losses = []
    params = train_policy(
        cfg, steps=args.steps, lr=args.lr, seed=args.seed,
        on_step=lambda i, l: losses.append(l),
    )
    print(f"trained {args.steps} steps in {time.time() - t0:.0f}s: "
          f"loss {losses[0]:.4f} -> {sum(losses[-50:]) / min(50, len(losses)):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_params(args.out, params, cfg)
    size_kb = args.out.stat().st_size / 1024
    print(f"saved {args.out} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
