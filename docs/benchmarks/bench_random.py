import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps, dtype):
    """Benchmark da.random.rand"""
    for i in range(reps):
        a = da.random.rand((size,size), dtype=dtype, comm=comm)

benchmark_function(f, 1000, 1000, dtype='float64')
benchmark_function(f, 2000, 1000, dtype='float64')
benchmark_function(f, 4000, 1000, dtype='float64')