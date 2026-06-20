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
    opt_parser.add_argument("--algorithm", type=str, default="MMA", help="Optimization algorithm (e.g. MMA, SLP, CONLIN)")
    opt_parser.add_argument("--use-line-search", action="store_true", help="Enable line search for the optimization algorithm")
    opt_parser.add_argument("--length", type=float, default=None, help="Domain length L")
    opt_parser.add_argument("--height", type=float, default=None, help="Domain height H")
    opt_parser.add_argument("--nelx", type=int, default=None, help="Number of elements in X direction")
    opt_parser.add_argument("--nely", type=int, default=None, help="Number of elements in Y direction")
    opt_parser.add_argument("--volfrac", type=float, default=0.4, help="Target volume fraction constraint")
    opt_parser.add_argument("--iterative", action="store_true", help="Use petsc4py iterative PCG solver instead of direct solver")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        from ggp.cli.runners.free_runner import run_main_ggp
        print(f"Running GGP Optimization ({args.formulation} formulation)...")
        run_main_ggp(
            bc_type=args.use_case, 
            max_iter=args.max_iter, 
            mode=args.formulation,
            algorithm=args.algorithm,
            use_line_search=args.use_line_search,
            L_opt=args.length,
            H_opt=args.height,
            nelx_opt=args.nelx,
            nely_opt=args.nely,
            volfrac_opt=args.volfrac,
            iterative=args.iterative
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
