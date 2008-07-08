import distarray as da
from benchmark import benchmark_function


def f(comm, size, reps):
    a = da.random.rand((size,size),comm=comm)
    b = da.random.rand((size,size),comm=comm)
    for i in range(reps):
        c = 10*a + 20*b

benchmark_function(f, 1000, 100)