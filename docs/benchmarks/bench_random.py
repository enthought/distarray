import distarray as da

def f(comm, size, reps):
    """Benchmark da.random.rand"""
    for i in range(reps):
        a = da.random.rand((size,size), comm=comm)

for size, reps in zip([1000,2000,4000],3*[10]):
    sizes, times, speedups = da.benchmark_function(f, size, reps)
    if da.COMM_PRIVATE.Get_rank()==0:
        print
        print "array_size, reps:", size, reps
        print sizes
        print times
        print speedups
