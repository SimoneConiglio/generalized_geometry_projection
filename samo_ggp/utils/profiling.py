import cProfile
import pstats
import io
import functools
import time
from pathlib import Path

def profile_me(output_file=None):
    """
    Decorator to profile a function using cProfile.
    Saves the stats to a .prof file and prints a summary.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            start_time = time.perf_counter()
            retval = prof.runcall(func, *args, **kwargs)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            print(f"\n[Profiler] Execution of {func.__name__} took {elapsed:.4f} seconds.")
            
            if output_file:
                prof.dump_stats(output_file)
                print(f"[Profiler] Detailed stats saved to {output_file}")
            
            # Print top 10 functions by cumulative time
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            print(s.getvalue())
            
            return retval
        return wrapper
    return decorator

class BenchmarkLogger:
    """
    Utility to record and compare timing results.
    """
    def __init__(self):
        self.results = {}

    def log_result(self, name, duration, iterations):
        self.results[name] = {
            "total_time": duration,
            "avg_per_iter": duration / iterations,
            "iterations": iterations
        }

    def print_summary(self):
        print("\n" + "="*40)
        print("     GGP ARCHITECTURE BENCHMARK")
        print("="*40)
        print(f"{'Implementation':<20} | {'Total (s)':<10} | {'Avg/Iter (s)':<12}")
        print("-" * 40)
        for name, data in self.results.items():
            print(f"{name:<20} | {data['total_time']:<10.3f} | {data['avg_per_iter']:<12.4f}")
        print("="*40)
