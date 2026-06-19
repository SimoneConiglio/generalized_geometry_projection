#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="GGP: Generalized Geometry Projection CLI",
        prog="ggp"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Optimization subcommand
    opt_parser = subparsers.add_parser("optimize", help="Run topology optimization")
    opt_parser.add_argument("--use-case", type=str, default="Short_Cantilever", 
                            choices=["Short_Cantilever", "MBB", "L-shape"], 
                            help="Boundary condition and domain")
    opt_parser.add_argument("--formulation", type=str, default="Free", 
                            choices=["Free", "ALM", "2D_Free", "3D_Free"], 
                            help="GGP formulation type")
    opt_parser.add_argument("--max-iter", type=int, default=50, help="Maximum number of outer optimization iterations")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        from ggp.cli.runners.free_runner import run_main_ggp
        print(f"Running GGP Optimization ({args.formulation} formulation)...")
        run_main_ggp(bc_type=args.use_case, max_iter=args.max_iter, mode=args.formulation)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
