import cProfile
import pstats
import io
import functools
import time
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

def get_git_revision_hash():
    """Returns the current git revision hash."""
    try:
        # Get root of git repo
        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii').strip()
        # Get hash
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode('ascii').strip()
    except Exception:
        return "unknown"

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
                # Ensure directory exists
                out_path = Path(output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                prof.dump_stats(str(out_path))
                print(f"[Profiler] Detailed stats saved to {output_file}")
            
            # Print top 10 functions by cumulative time
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            print(s.getvalue())
            
            return retval
        return wrapper
    return decorator

class PerformanceTracker:
    """
    Utility to record performance metrics and track evolution in a JSON file.
    """
    def __init__(self, history_file="performance_history.json"):
        self.history_file = history_file
        self.current_session = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_revision_hash(),
            "benchmarks": {}
        }

    def log_benchmark(self, name, duration, iterations, metadata=None):
        """Logs a benchmark result to the current session."""
        avg_time = duration / iterations if iterations > 0 else 0
        self.current_session["benchmarks"][name] = {
            "total_time": duration,
            "avg_per_iter": avg_time,
            "iterations": iterations,
            "metadata": metadata or {}
        }
        print(f"[PerformanceTracker] Logged {name}: {avg_time:.4f} s/iter")

    def save(self):
        """Saves the current session results to the history file."""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        
        history.append(self.current_session)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[PerformanceTracker] History saved to {self.history_file}")

    def print_summary(self):
        """Prints a summary of the current session's benchmarks."""
        print("\n" + "="*65)
        print(f"     PERFORMANCE SUMMARY ({self.current_session['timestamp']})")
        print("="*65)
        print(f"{'Benchmark':<25} | {'Avg/Iter (s)':<15} | {'Iterations':<10}")
        print("-" * 65)
        for name, data in self.current_session["benchmarks"].items():
            print(f"{name:<25} | {data['avg_per_iter']:<15.4f} | {data['iterations']:<10}")
        print("="*65)
