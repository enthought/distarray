import pylab

ideal = [1,2,4,8,16]
ncpus = [1,2,4,8,16]
multicore = [1,1.9921,4.11310,6.77]
cluster = [1,1.9692,2.84175,3.0178,5.2564]

pylab.plot(ncpus, ideal,'r-', label='ideal')
pylab.plot(ncpus[:-1], multicore, 'g-o', label='multicore')
pylab.plot(ncpus, cluster, 'b-o', label='gigE cluster')

pylab.xlabel('Number of processors')
pylab.ylabel('Speedup')
pylab.title('Parallel Scaling for FFT')

pylab.legend(loc=2)
