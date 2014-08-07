# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate pi using a Monte Carlo method using pure NumPy.
"""

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


def main(N):
    result, time = calc_pi(N)
    print('time  : %3.4g\nresult: %.7f' % (time, result))


if __name__ == '__main__':
    import argparse
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("npoints", metavar="N", type=int,
                        help=("number of points to use in estimation"))
    args = parser.parse_args()
    main(args.npoints)
