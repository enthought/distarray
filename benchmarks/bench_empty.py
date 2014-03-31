# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------


from __future__ import print_function
import distarray as da
from benchmark import benchmark_function


def f(comm, size, reps, dtype):
    """Benchmark da.empty"""
    for i in range(reps):
        a = da.empty((size,size), dtype=dtype, comm=comm)

for size, reps in zip([1000,2000,4000],3*[10]):
    sizes, times, speedups = benchmark_function(f, size, reps, 'float64')
    if da.mpiutils.COMM_PRIVATE.Get_rank()==0:
        print()
        print("array_size, reps:", size, reps)
        print(sizes)
        print(times)
