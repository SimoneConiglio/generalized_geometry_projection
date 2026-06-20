import pytest
import time
import os
from ggp.utils.profiling import Profiler, Timer, profile_function, print_profiler_stats, export_profiler_stats

def test_profiler_timer():
    Profiler.clear()
    with Timer("test_block"):
        time.sleep(0.01)
        
    stats = Profiler.get_stats()
    assert "test_block" in stats
    assert stats["test_block"]["calls"] == 1
    assert stats["test_block"]["total_time"] >= 0.01

def test_profile_function_decorator():
    Profiler.clear()
    
    @profile_function("my_func")
    def dummy_func():
        time.sleep(0.01)
        return True
        
    assert dummy_func() is True
    stats = Profiler.get_stats()
    assert "my_func" in stats
    assert stats["my_func"]["calls"] == 1
    assert stats["my_func"]["total_time"] >= 0.01

def test_profiler_printing_and_export(tmpdir):
    Profiler.clear()
    with Timer("print_test"):
        time.sleep(0.01)
        
    # Should not crash
    print_profiler_stats()
    
    out_file = os.path.join(str(tmpdir), "stats.json")
    export_profiler_stats(out_file)
    assert os.path.exists(out_file)
