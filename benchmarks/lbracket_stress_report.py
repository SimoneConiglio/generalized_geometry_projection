# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Merge the local-stress-constraint solver benchmark into a table and figures.

Sources: ``lbracket_stress_solvers_results.jsonl`` (IPOPT, NLopt, ParOpt, AL + MMA,
TAO ALMM) and ``lbracket_density_uno_*.json`` (Uno presets). Quality metric: the best
volume among iterates with max g <= 1 % (``TOL``), from each run's history.

    python -m benchmarks.lbracket_stress_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
STATIC = OUT.parent / "docs" / "_static"
TOL = 0.01
LABEL = {
    "ipopt": "IPOPT", "nlopt-mma": "NLopt MMA", "paropt-mma": "ParOpt MMA",
    "paropt-tr": "ParOpt TR", "al-mma": "AL + MMA", "tao-almm": "TAO ALMM",
    "uno-filtersqp": "Uno filterSQP", "uno-ipopt": "Uno IPM", "uno-filterslp": "Uno filterSLP",
    "uno-funnelsqp": "Uno funnelSQP",
}
JACOBIAN_FREE = {"al-mma", "tao-almm"}


def best_within(hist, tol=TOL):
    ok = [(t, v) for t, v, g in hist if g <= tol]
    if not ok:
        return float("nan"), float("nan")
    i = int(np.argmin([v for _, v in ok]))
    return ok[i][1], ok[i][0]


def load_rows():
    rows = {}
    path = OUT / "lbracket_stress_solvers_results.jsonl"
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            rows[(r["method"], r["n"])] = r
    for path in sorted(OUT.glob("lbracket_density_uno_*.json")):
        for r in json.load(open(path)):
            method = f"uno-{r['preset']}"
            rows[(method, r["n"])] = {
                "method": method, "n": r["n"], "nelx": r["nelx"],
                "status": r["status"], "iterations": r["iterations"],
                "volume": r["volume"], "max_g": r["max_g"],
                "n_fe": r["n_constraint_evals"], "n_rhs": r["n_jacobian_evals"] * r["n"],
                "wall_s": r["wall_s"], "fe_s": r["fe_s"], "opt_s": r["uno_s"],
                "peak_rss_mb": None, "hist": r["hist"], "x": r["x"],
                "options": r.get("options") or {},
            }
    out = []
    for r in rows.values():
        if "hist" in r:
            r["best_v"], r["t_best"] = best_within(r["hist"])
        out.append(r)
    return sorted(out, key=lambda r: (r["n"], r["method"]))


def report():
    rows = load_rows()
    lines = [
        "# Open-source optimizers on local stress constraints (density L-bracket)",
        "",
        f"Best V = lowest volume among iterates with max g <= {TOL:g} (1 % stress overshoot).",
        "Time split: FE = primal solves + adjoint/forward back-substitutions + Jacobian;",
        "opt = the optimizer's own time. Regenerate: `python -m benchmarks.lbracket_stress_report`.",
        "",
        "| n = m | Method | Best V (max g <= 1 %) | Final V | Final max g | FE solves "
        "| Back-substitutions | Wall (s) | FE (s) | Optimizer (s) | Peak memory (MB) | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "volume" not in r:
            lines.append(f"| {r['n']} | {LABEL.get(r['method'], r['method'])} | - | - | - | - "
                         f"| - | - | - | - | - | {r['status']} |")
            continue
        mem = f"{r['peak_rss_mb']:.0f}" if r.get("peak_rss_mb") else "-"
        lines.append(
            f"| {r['n']} | {LABEL.get(r['method'], r['method'])} | {r['best_v']:.4f} | "
            f"{r['volume']:.4f} | {r['max_g']:+.1e} | {r['n_fe']} | {r['n_rhs']:.2e} | "
            f"{r['wall_s']:.0f} | {r['fe_s']:.0f} | {r['opt_s']:.0f} | {mem} | "
            f"{str(r['status'])[:40]} |")
    (OUT / "lbracket_stress_results.md").write_text("\n".join(lines) + "\n")

    # compact rows for charts (one per method and size)
    compact = [{k: r.get(k) for k in ("method", "n", "best_v", "volume", "max_g", "wall_s",
                                       "fe_s", "opt_s", "n_fe", "n_rhs", "peak_rss_mb",
                                       "status")} for r in rows]
    json.dump(compact, open(OUT / "lbracket_stress_summary.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows if "volume" in r})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in methods:
        rs = [r for r in rows if r["method"] == m and "volume" in r]
        n = [r["n"] for r in rs]
        ls = "-" if m in JACOBIAN_FREE else "--"
        axes[0].loglog(n, [r["wall_s"] for r in rs], "o" + ls, label=LABEL.get(m, m))
        axes[1].semilogx(n, [r["best_v"] for r in rs], "o" + ls, label=LABEL.get(m, m))
    axes[0].set_xlabel("n = m")
    axes[0].set_ylabel("wall time (s)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("n = m")
    axes[1].set_ylabel("best volume with max g <= 1 %")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STATIC / "lbracket_stress_solvers.png", dpi=130)


if __name__ == "__main__":
    report()
