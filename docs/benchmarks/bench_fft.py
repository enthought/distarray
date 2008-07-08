import distarray as da

def f(comm, size, reps):
    """Benchmark da.fft"""
    a = da.random.rand((size,size), comm=comm)
    b = a.astype(newdtype='complex128')
    filter = da.random.rand((size, size), comm=comm)
    for i in range(reps):
        c = da.fft.fft2(b)
        d = c*filter
        e = da.fftw.ifft2(c)

for size, reps in zip([1024,2048,4096],3*[1]):
    sizes, times = da.benchmark_function(f, size, reps)
    if da.COMM_PRIVATE.Get_rank()==0:
        print
        print "array_size, reps:", size, reps
        print sizes
        print times