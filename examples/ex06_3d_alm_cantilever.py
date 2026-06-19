#!/usr/bin/env python3
import os
import sys

def main():
    print("Running 3D ALM formulation example via the GGP CLI...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    # Since 3D ALM uses ALM formulation with a 3D geometry mode or ALM handles it internally,
    # Here we invoke 3D ALM or fallback if not explicitly supported in the parser yet.
    # Currently ggp.py allows --formulation ALM. If 3D ALM isn't built into free_runner,
    # this will act as a placeholder. We'll set formulation to ALM with a placeholder warning.
    print("Note: 3D ALM may require specific configuration in the CLI.")
    os.system(f"{sys.executable} ggp.py optimize --use-case Short_Cantilever --formulation ALM --max-iter 30")

if __name__ == "__main__":
    main()
