import distarray as da
from benchmark import benchmark_function


def f(comm):
    a = da.random.rand((1000,1000),comm=comm)
    b = da.random.rand((1000,1000),comm=comm)
    for i in range(100):
        c = 10*a + 20*b

benchmark_function(f)