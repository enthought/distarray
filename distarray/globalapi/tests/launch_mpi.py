"""
Script to test launching an MPI-only client.

    $ mpiexec -np <np> python launch_mpi.py

If exits cleanly, then everything is fine.  If exits with an error code, then
there's a problem.

"""

from __future__ import print_function
from distarray.globalapi import Context, Distribution
import numpy as np

c = Context(kind='MPI')

fmt = lambda s: "{:.<25s}:".format(s)

print(fmt("Context"), c)
print(fmt("targets"), c.targets)

if __name__ == '__main__':
    size = len(c.targets) * 100
    print(fmt("size"), size)
    dist = Distribution(c, (size,))
    print(fmt("Distribution"), dist)
    da = c.ones(dist, dtype=np.int64)
    print(fmt("DistArray"), da)
    factor = 2
    db = da * factor
    print(fmt("DistArray"), db)
    sum = db.sum().tondarray()
    print(fmt("sum"), sum)
    print(fmt("sum == factor * size"), sum == size * factor)
    assert sum == size * factor
