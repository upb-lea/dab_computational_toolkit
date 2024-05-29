"""Different methods to debug this toolbox."""
import os
from datetime import datetime
# Decorator to print function args
import inspect
# Decorator to measure execution time of a function
from functools import wraps
import time

# Set a global DEBUG variable to switch some debugging code.
# This is evaluated a runtime, not like the Python __debug__ that is evaluated in preprocess.
DEBUG = False


class log:
    """Class to print logging text to stdout and to a log file."""

    logfile = None

    def __init__(self, filename=str()):
        """Init the log file."""
        if filename:
            filename = os.path.expanduser(filename)
            filename = os.path.expandvars(filename)
            filename = os.path.abspath(filename)
            self.logfile = open(filename, 'a', buffering=1)

    def __del__(self):
        """Close the log file."""
        self.close()

    def close(self):
        """Close the log file."""
        if self.logfile:
            self.logfile.close()

    def error(self, *args, sep='\n', **kwargs):
        r"""
        Log error output like print does.

        :param sep: separator definition. Default to '\n'
        """
        # print(*args, **kwargs)
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
            inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs)
        if self.logfile:
            print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
                inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs, file=self.logfile)

    def warning(self, *args, sep='\n', **kwargs):
        r"""
        Log warning output like print does.

        :param sep: separator definition. Default to '\n'
        """
        # print(*args, **kwargs)
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
            inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs)
        if self.logfile:
            print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
                inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs, file=self.logfile)

    def info(self, *args, sep='\n', **kwargs):
        r"""
        Log normal info output like print does.

        :param sep: separator definition. Default to '\n'
        """
        print(*args, **kwargs, sep=sep)
        if self.logfile:
            print(*args, **kwargs, sep=sep, file=self.logfile)

    def debug(self, *args, sep='\n', **kwargs):
        r"""
        Log debug output like print does.

        :param sep: separator definition. Default to '\n'
        """
        if DEBUG or __debug__:
            # highly detailed output
            print(datetime.now().isoformat(timespec='milliseconds') + ' ' + \
                  inspect.getmodule(inspect.stack()[1][0]).__name__ + ' ' + \
                  inspect.currentframe().f_back.f_code.co_name + '\n' + \
                  sep.join(map(str, args)), **kwargs)
            if self.logfile:
                print(datetime.now().isoformat(timespec='milliseconds') + ' ' + \
                      inspect.getmodule(inspect.stack()[1][0]).__name__ + ' ' + \
                      inspect.currentframe().f_back.f_code.co_name + '\n' + \
                      sep.join(map(str, args)), **kwargs, file=self.logfile)


def dump_args(func):
    """
    Print function call details by the use of an oparator.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


# Use this in front of a function to print args
# @dump_args


def timeit(func):
    """Measure execution time of a function by the use of a decorator."""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        # print(f'{total_time:.4f} seconds for Function {func.__name__}{args} {kwargs}')
        print(f'{total_time:.4f} seconds for Function {func.__name__}')
        return result

    return timeit_wrapper


# Use this in front of a function to measure execution time
# @timeit


def error(*args, sep='\n', **kwargs):
    """
    Log error output like print does.

    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ + \
          ' ' + sep.join(map(str, args)), **kwargs)


def warning(*args, sep='\n', **kwargs):
    """
    Log warning output like print does.

    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ + \
          ' ' + sep.join(map(str, args)), **kwargs)


def info(*args, sep='\n', **kwargs):
    """
    Log normal info output like print does.

    :param args:
    :param sep:
    :param kwargs:
    """
    print(*args, **kwargs)
    # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + sep.join(map(str, args)), **kwargs)
    # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
    #       ' ' + sep.join(map(str, args)), **kwargs)


def debug(*args, sep='\n', **kwargs):
    r"""
    Log debug output like print does.

    :param args: arguments
    :param sep: separator, default is '\n'
    :param kwargs:
    """
    if DEBUG or __debug__:
        # print(*args, **kwargs)
        # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + sep.join(map(str, args)), **kwargs)
        # detailed output
        # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
        #       ' ' + sep.join(map(str, args)), **kwargs)
        # highly detailed output
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' + \
              inspect.getmodule(inspect.stack()[1][0]).__name__ + ' ' + \
              inspect.currentframe().f_back.f_code.co_name + '\n' + \
              sep.join(map(str, args)), **kwargs)


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module Debug ...")
