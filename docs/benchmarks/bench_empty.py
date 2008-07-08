import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps, dtype):
    """Benchmark da.empty"""
    for i in range(reps):
        a = da.empty((size,size), dtype=dtype, comm=comm)

benchmark_function(f, 1000, 1000, dtype='float64')
benchmark_function(f, 2000, 1000, dtype='float64')
benchmark_function(f, 4000, 1000, dtype='float64')

