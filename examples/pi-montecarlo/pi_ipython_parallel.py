"""
Calculate pi using a Monte Carlo method using IPython Parallel.
Usage:
    python pi_ipython_parallel.py <number of points>
"""

import sys
from IPython.parallel import Client, interactive

client = Client()
view = client[:]
view.execute('import numpy')


@interactive
def calc_pi_on_engines(n):
    x = numpy.random.rand(n)
    y = numpy.random.rand(n)
    r = numpy.hypot(x, y)
    return 4. * (r < 1.).sum() / n


def calc_pi(n):
    n_engines = n/len(view)
    results = view.apply_sync(calc_pi_on_engines, n_engines)
    client.purge_everything()
    return float(sum(results))/len(results)

if __name__ == "__main__":
    print(calc_pi(int(sys.argv[1])))
