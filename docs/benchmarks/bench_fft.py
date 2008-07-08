import distarray as da

def f(comm, size, reps):
    """Benchmark da.fft"""
    a = da.random.rand((size,size), comm=comm)
    b = a.astype(newdtype='complex128')
    for i in range(reps):
        c = da.fft.fft2(b)
        # d = da.fftw.ifft2(c)

for size, reps in zip([1000,2000,4000],3*[10]):
    sizes, times = da.benchmark_function(f, size, reps)
    if len(sizes)==3:
        print
        print "array_size, reps:", size, reps
        print sizes
        print times