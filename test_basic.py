'''
To run this file, you need to start a cluster with the following command:

    $ ipcluster start -n <n> --engines=MPIEngineSetLauncher
'''


import numpy as np
import os
from IPython.parallel import Client
from distarray.client import DistArrayContext
from timeit import timeit

c = Client()
dv = c[:]
dv.execute('import os; os.chdir("%s")' % os.path.abspath(os.getcwd()))
dac = DistArrayContext(dv)
dist_timings = []
np_timings = []
number = 3
for p in range(5, 27):
    print p
    N = 2**p
    dist_a = dac.empty((N,))
    reg_a = np.empty((N,))
    # dist_timings.append((N, timeit('dac.sin(dist_a)', setup='from __main__ import dac, dist_a', number=number) / float(number)))
    # np_timings.append((N, timeit('np.sin(reg_a)', setup='from __main__ import np, reg_a', number=number) / float(number)))
    dist_timings.append((N, timeit('dac.add(dist_a, dist_a)', setup='from __main__ import dac, dist_a', number=number) / float(number)))
    np_timings.append((N, timeit('np.add(reg_a, reg_a)', setup='from __main__ import np, reg_a', number=number) / float(number)))

np_timings = np.array(np_timings)
dist_timings = np.array(dist_timings)
