# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate pi using a Monte Carlo method using IPython Parallel.
Usage:
    python pi_ipython_parallel.py <number of points>
"""

import sys
from IPython.parallel import Client, interactive

from util import timer

client = Client()
view = client[:]
view.execute('import numpy')


@interactive  # this runs on the engins
def calc_pi_on_engines(n):
    x = numpy.random.rand(n)
    y = numpy.random.rand(n)
    r = numpy.hypot(x, y)
    return 4. * (r < 1.).sum() / n


@timer
def calc_pi(n):
    """Estimate pi using IPython.parallel."""
    n_engines = n/len(view)
    results = view.apply_sync(calc_pi_on_engines, n_engines)
    return float(sum(results))/len(results)

if __name__ == "__main__":
    N = int(sys.argv[1])
    result, time = calc_pi(N)
    print('time  : %3.4g\nresult: %.7f' % (time, result))
    client.purge_everything()
