# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Estimate pi using a Monte Carlo method with distarray.
Usage:
    $ python pi_distarray.py <number of points>
"""

import sys

import distarray
from distarray.random import Random

from util import timer

context = distarray.Context()
random = Random(context)


@timer
def calc_pi(n):
    """Estimate pi using distributed NumPy arrays."""
    x = random.rand((n,))
    y = random.rand((n,))
    r = distarray.hypot(x, y)
    return 4 * float((r < 1.).sum())/n

if __name__ == '__main__':
    N = int(sys.argv[1])
    result, time = calc_pi(N)
    print('time  : %3.4g\nresult: %.7f' % (time, result))
    context.view.client.purge_everything()
