"""
Calculate pi using a Monte Carlo method using pure NumPy.
Usage:
    python pi_numpy.py <number of points>
"""

import sys
import numpy
from numpy import random

from util import timer

@timer
def calc_pi(n):
    """Estimate pi using pure NumPy."""
    x = random.rand(n)
    y = random.rand(n)
    r = numpy.hypot(x, y)
    return 4 * float((r < 1.).sum()) / n

if __name__ == '__main__':
    # Get the number of points.
    N = int(sys.argv[1])
    result, time = calc_pi(N)
    print('time  : %3.4g\nresult: %.7f' % (time, result))
