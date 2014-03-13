# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------


import pylab

ideal = [1,2,4,8,16]
ncpus = [1,2,4,8,16]
multicore = [1,1.441,4.139,0.3647]
cluster = [1,1.976,3.827,7.15744,12.42]

pylab.plot(ncpus, ideal,'r-', label='ideal')
pylab.plot(ncpus[:-1], multicore, 'g-o', label='multicore')
pylab.plot(ncpus, cluster, 'b-o', label='gigE cluster')

pylab.xlabel('Number of processors')
pylab.ylabel('Speedup')
pylab.title('Parallel Scaling for Basic Array Operations')

pylab.legend(loc=2)
