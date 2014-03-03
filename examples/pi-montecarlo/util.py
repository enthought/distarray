from timeit import default_timer as clock
from functools import wraps


def timer(fn):
    @wraps(fn)
    def _timer(*args, **kwargs):
        start = clock()
        result = fn(*args, **kwargs)
        stop = clock()
        return (result, stop - start)
    return _timer
