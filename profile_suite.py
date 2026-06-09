import time
import os
import sys
from samo_ggp.utils.profiling import profile_me, PerformanceTracker

# Add current directory to path for imports
sys.path.append(os.getcwd())

# Import example runners
from examples import (
    ex01_short_cantilever as short_cantilever, 
    ex02_mbb_beam as mbb_beam, 
    ex03_l_shape_bracket as l_shape_bracket, 
    ex04_alm_cantilever as alm_cantilever
)

def run_suite(iterations=5):
    tracker = PerformanceTracker()
    
    benchmarks = [
        ("Short Cantilever", short_cantilever.run_short_cantilever),
        ("MBB Beam", mbb_beam.run_mbb_beam),
        ("L-Shape Bracket", l_shape_bracket.run_l_shape_bracket),
        ("ALM Cantilever", alm_cantilever.run_alm_cantilever),
    ]
    
    for name, func in benchmarks:
        print(f"\n>>> Profiling {name} ({iterations} iterations)...")
        
        # Define a wrapped function for cProfile
        prof_file = f"performance_logs/{name.lower().replace(' ', '_')}.prof"
        
        @profile_me(output_file=prof_file)
        def timed_run():
            start = time.perf_counter()
            func(max_iter=iterations)
            return time.perf_counter() - start
            
        duration = timed_run()
        tracker.log_benchmark(name, duration, iterations)

    tracker.print_summary()
    tracker.save()

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("performance_logs", exist_ok=True)
    
    # Run suite with 2 iterations for fast profiling
    run_suite(iterations=2)
