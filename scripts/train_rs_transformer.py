# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Train the TRANSFORMER_2D reduced-space policy and save its weights.

Training is behaviour cloning of a grid teacher that solves the true 2D
trust-region subproblem, rolled through the actual reduced-space driver on
**generic synthetic families only** (free / linearly-constrained /
ball-constrained quadratics, curved valleys; dimensions 4..256). No
topology-optimization data is used: the toy-SIMP family and GGP problems are
held out for testing generalization.

Usage (inside the ``ggp`` conda environment)::

    python scripts/train_rs_transformer.py --steps 3000 \
        --out scp_uno/weights/rs_transformer_default.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scp_uno.rs_transformer import RSPolicyConfig, save_params, train_policy

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-states", type=int, default=64)
    parser.add_argument("--out", type=Path,
                        default=REPO / "scp_uno" / "weights" / "rs_transformer_default.npz")
    args = parser.parse_args()

    cfg = RSPolicyConfig()
    t0 = time.time()
    losses = []
    params = train_policy(cfg, steps=args.steps, lr=args.lr,
                          batch_states=args.batch_states, seed=args.seed,
                          on_step=lambda i, l: losses.append(l))
    print(f"trained {args.steps} steps in {time.time() - t0:.0f}s: "
          f"loss {sum(losses[:10]) / 10:.4f} -> {sum(losses[-50:]) / 50:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_params(args.out, params, cfg)
    print(f"saved {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
