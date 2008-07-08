import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps, dtype):
    """Benchmark da.ones"""
    for i in range(reps):
        a = da.ones((size,size), dtype=dtype, comm=comm)

benchmark_function(f, 1000, 1000, dtype='float64')
benchmark_function(f, 2000, 1000, dtype='float64')
benchmark_function(f, 4000, 1000, dtype='float64')