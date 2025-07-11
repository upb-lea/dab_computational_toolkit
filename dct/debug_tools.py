"""Different methods to debug this toolbox."""
# Decorator to print function args
# Decorator to measure execution time of a function
from functools import wraps
import time

# Set a global DEBUG variable to switch some debugging code.
# This is evaluated a runtime, not like the Python __debug__ that is evaluated in a pre-process.
DEBUG = False

def timeit(func):
    """
    Measure execution time of a function by the use of a decorator.

    Use this in front of a function to measure execution time
    @timeit

    :param func: function to stop the time
    :type func: python function
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{total_time:.4f} seconds for Function {func.__name__}')
        return result

    return timeit_wrapper
