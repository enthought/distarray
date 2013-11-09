from __future__ import print_function
import distarray as da
from benchmark import benchmark_function


def f(comm, size, reps):
    """Benchmark da.add"""
    a = da.random.rand((size,size), comm=comm)
    b = da.random.rand((size,size), comm=comm)
    c = da.empty_like(a)
    for i in range(reps):
        da.add(a,b,c)


for size, reps in zip([1000,2000,4000],3*[10]):
    sizes, times, speedups = benchmark_function(f, size, reps)
    if da.mpiutils.COMM_PRIVATE.Get_rank()==0:
        print()
        print("array_size, reps:", size, reps)
        print(sizes)
        print(times)
        print(speedups)
