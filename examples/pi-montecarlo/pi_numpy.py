"""
Calculate pi using a Monte Carlo method using pure NumPy.
Usage:
    python pi_numpy.py <number of points>
"""

import sys
import numpy


def calc_pi(n):
    """Estimate pi"""
    x = numpy.random.rand(n)
    y = numpy.random.rand(n)
    r = numpy.hypot(x, y)
    return 4 * float((r < 1.).sum()) / n

if __name__ == '__main__':
    # Get the number of points.
    N = int(sys.argv[1])
    print(calc_pi(N))
