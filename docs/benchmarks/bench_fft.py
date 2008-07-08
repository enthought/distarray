import distarray as da
from benchmark import benchmark_function

def f(comm, size, reps):
    """Benchmark da.fft"""
    a = da.random.rand((size,size), comm=comm)
    b = a.astype(newdtype='complex128')
    for i in range(reps):
        c = da.fft.fft2(b)
        # d = da.fftw.ifft2(c)

benchmark_function(f, 1000, 10)
benchmark_function(f, 2000, 10)
benchmark_function(f, 4000, 10)
# benchmark_function(f, 8000, 100)