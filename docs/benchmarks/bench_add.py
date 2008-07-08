import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps):
    """Benchmark da.add"""
    a = da.random.rand((size,size), comm=comm)
    b = da.random.rand((size,size), comm=comm)
    c = da.empty_like(a)
    for i in range(reps):
        da.add(a,b,c)

benchmark_function(f, 1000, 500)
benchmark_function(f, 2000, 500)
benchmark_function(f, 4000, 500)