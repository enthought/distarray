"""
Estimate pi using a Monte Carlo method with distarray.
Usage:
    $ python pi_distarray.py <number of points>
"""
import sys
from distarray.client import RandomModule, Context

context = Context()
random = RandomModule(context)


def calc_pi(n):
    x = random.rand((n,))
    y = random.rand((n,))
    r = context.hypot(x, y)
    pi = 4 * float((r < 1.).sum())/n
    context.view.client.purge_everything()
    return pi

if __name__ == '__main__':
    N = int(sys.argv[1])
    print(calc_pi(N))
