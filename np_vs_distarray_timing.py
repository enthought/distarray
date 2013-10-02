'''
To run this file, you need to start a cluster with the following command:

    $ ipcluster start -n <n> --engines=MPIEngineSetLauncher
'''

import numpy as np
import os
from IPython.parallel import Client
from distarray.client import Context
from timeit import timeit

c = Client()
dv = c[:]
dac = Context(dv)

def run_timings():

    global dist_a, reg_a

    dist_timings = []
    np_timings = []
    repeat = 3
    for p in range(5, 10):
        print p
        N = 2**p
        dist_a = dac.empty((N,))
        reg_a = np.empty((N,))
        dist_timings.append((N, timeit('dac.add(dist_a, dist_a)',
                                    setup='from __main__ import dac, dist_a',
                                    number=repeat) / float(repeat)))
        np_timings.append((N, timeit('np.add(reg_a, reg_a)',
                                    setup='from __main__ import np, reg_a',
                                    number=repeat) / float(repeat)))

    np_timings = np.array(np_timings)
    dist_timings = np.array(dist_timings)
    return np_timings, dist_timings

if __name__ == '__main__':
    np_timings, dist_timings = run_timings()
