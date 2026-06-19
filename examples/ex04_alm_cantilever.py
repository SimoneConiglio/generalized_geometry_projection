#!/usr/bin/env python3
import os
import sys

def main():
    print("Running ALM Cantilever with continuous Overhang Constraints via the GGP CLI...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.system(f"{sys.executable} ggp.py optimize --use-case Short_Cantilever --formulation ALM --max-iter 30")

if __name__ == "__main__":
    main()
