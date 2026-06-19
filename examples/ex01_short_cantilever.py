#!/usr/bin/env python3
import os
import sys

def main():
    print("Running Short Cantilever example via the GGP CLI...")
    # Navigate to the project root to invoke the CLI properly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.system(f"{sys.executable} ggp.py optimize --use-case Short_Cantilever --formulation Free --max-iter 50")

if __name__ == "__main__":
    main()
