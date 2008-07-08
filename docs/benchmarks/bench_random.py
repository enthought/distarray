import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps):
    """Benchmark da.random.rand"""
    for i in range(reps):
        a = da.random.rand((size,size), comm=comm)

benchmark_function(f, 1000, 1000)
benchmark_function(f, 2000, 1000)
benchmark_function(f, 4000, 1000)