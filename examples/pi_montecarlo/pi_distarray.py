# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Estimate pi using a Monte Carlo method with distarray.
"""

from __future__ import division, print_function

from util import timer

from distarray.globalapi import Context, Distribution, hypot
from distarray.globalapi.random import Random


context = Context()
random = Random(context)


@timer
def calc_pi(n):
    """Estimate pi using distributed NumPy arrays."""
    distribution = Distribution(context=context, shape=(n,))
    x = random.rand(distribution)
    y = random.rand(distribution)
    r = hypot(x, y)
    mask = (r < 1)
    return 4 * mask.sum().toarray() / n


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
