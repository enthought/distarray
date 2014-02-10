"""
Estimate pi using a Monte Carlo method with distarray.
Usage:
    $ python pi_distarray.py <number of points>
"""
import sys
from distarray.client import RandomModule, Context

from util import timer

context = Context()
random = RandomModule(context)

@timer
def calc_pi(n):
    """Estimate pi using distributed NumPy arrays."""
    x = random.rand((n,))
    y = random.rand((n,))
    r = context.hypot(x, y)
    return 4 * float((r < 1.).sum())/n

if __name__ == '__main__':
    N = int(sys.argv[1])
    result, time = calc_pi(N)
    print('time  : %3.4g\nresult: %.7f' % (time, result))
    context.view.client.purge_everything()
