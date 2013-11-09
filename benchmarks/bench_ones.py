from __future__ import print_function
import distarray as da
from benchmark import benchmark_function


def f(comm, size, reps, dtype):
    """Benchmark da.ones"""
    for i in range(reps):
        a = da.ones((size,size), dtype=dtype, comm=comm)


for size, reps in zip([1000,2000,4000],3*[10]):
    sizes, times, speedups = benchmark_function(f, size, reps, 'float64')
    if da.mpiutils.COMM_PRIVATE.Get_rank()==0:
        print()
        print("array_size, reps:", size, reps)
        print(sizes)
        print(times)
