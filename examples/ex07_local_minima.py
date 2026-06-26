#!/usr/bin/env python3
"""Example 07 — improved local minima on the Short Cantilever 2D.

Runs the SC 2D benchmark that compares the single-start baseline against the four
global-search strategies (continuation, multi-start, basin hopping, deflation).
Defaults to the fast *reduced* sweep; pass ``--full`` for publication fidelity.
"""
import os
import sys


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    extra = " ".join(sys.argv[1:])
    os.system(f"{sys.executable} benchmarks/sc2d_local_minima.py {extra}")


if __name__ == "__main__":
    main()
