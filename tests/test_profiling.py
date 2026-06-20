import pytest
import os
import time
from ggp.utils.profiling import PerformanceTracker, profile_me, get_git_revision_hash

def test_get_git_revision_hash():
    # Should not crash, returns string
    h = get_git_revision_hash()
    assert isinstance(h, str)

def test_performance_tracker(tmpdir):
    history_file = os.path.join(str(tmpdir), "test_history.json")
    tracker = PerformanceTracker(history_file)
    
    # Log some dummy benchmarks
    tracker.log_benchmark("test_bench_1", 0.5, 10)
    tracker.log_benchmark("test_bench_2", 0.1, 5)
    
    # Save history
    tracker.save()
    assert os.path.exists(history_file)
    
    # Load again to test append
    tracker2 = PerformanceTracker(history_file)
    tracker2.log_benchmark("test_bench_3", 0.2, 2)
    tracker2.save()
    
    tracker2.print_summary()

def test_profile_me_decorator(tmpdir):
    prof_out = os.path.join(str(tmpdir), "test.prof")
    
    @profile_me(output_file=prof_out)
    def dummy_func():
        time.sleep(0.01)
        return 42
        
    res = dummy_func()
    assert res == 42
    assert os.path.exists(prof_out)
