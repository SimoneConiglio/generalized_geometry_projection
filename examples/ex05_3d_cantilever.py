#!/usr/bin/env python3
import os
import sys

def main():
    print("Running 3D Free formulation example via the GGP CLI...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.system(f"{sys.executable} ggp.py optimize --use-case Short_Cantilever --formulation 3D_Free --max-iter 30 --iterative")

if __name__ == "__main__":
    main()
