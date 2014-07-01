# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate the Julia set for a given z <- z**2 + c with
distarray passed via command line args. Usage:
    $ python julia_distarray.py <c real component> <c imaginary component>
"""

import sys
from time import time
from matplotlib import pyplot

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray
from distarray.dist.decorators import local


context = Context()
with context.view.sync_imports():
    import numpy


def create_complex_plane(context, resolution, dist, re_ax, im_ax):
    ''' Create a DistArray containing points on the complex plane.

    resolution: A 2-tuple with the number of points along Re and Im axes.
    dist: Distribution for the DistArray.
    re_ax: A 2-tuple with the (lower, upper) range of the Re axis.
    im_ax: A 2-tuple with the (lower, upper) range of the Im axis.
    '''

    @local
    def fill_complex_plane(arr, re_ax, im_ax, resolution):
        ''' Fill in points on the complex coordinate plane. '''
        # Drawing the coordinate plane directly like this is currently much
        # faster than trying to do it by indexing a distarray.
        # This may not be the most DistArray-thonic way to do this.
        re_step = float(re_ax[1] - re_ax[0]) / resolution[0]
        im_step = float(im_ax[1] - im_ax[0]) / resolution[1]
        for i in arr.distribution[0].global_iter:
            for j in arr.distribution[1].global_iter:
                arr.global_index[i, j] = complex(re_ax[0] + re_step * i,
                                                 im_ax[0] + im_step * j)

    # Create an empty distributed array.
    distribution = Distribution.from_shape(context,
                                           (resolution[0], resolution[1]),
                                           dist=dist)
    complex_plane = context.empty(distribution, dtype=complex)
    fill_complex_plane(complex_plane, re_ax, im_ax, resolution)
    return complex_plane


def local_julia_calc(la, c, z_max, n_max):
    ''' Calculate the number of iterations for the point to escape.

    la: Local array of complex values whose iterations we will count.
    c: Complex number to add at each iteration.
    z_max: Magnitude of complex value that we assume goes to infinity.
    n_max: Maximum number of iterations.
    '''

    @numpy.vectorize
    def julia_calc(z, c, z_max, n_max):
        ''' Use usual numpy.vectorize to apply on all the complex points. '''
        n = 0
        while abs(z) < z_max and n < n_max:
            z = z * z + c
            n += 1
        return n

    from distarray.local import LocalArray
    a = la.ndarray
    b = julia_calc(a, c, z_max, n_max)
    res = LocalArray(la.distribution, buf=b)
    rtn = proxyize(res)
    return rtn


def distributed_julia_calc(distarray, c, z_max=10, n_max=100):
    ''' Calculate the Julia set for an array of points in the complex plane.

    distarray: DistArray of complex values whose iterations we will count.
    c: Complex number to add at each iteration.
    z_max: Magnitude of complex value that we assume goes to infinity.
    n_max: Maximum number of iterations.
    '''
    context = distarray.context
    iters_key = context.apply(local_julia_calc,
                              (distarray.key, c, z_max, n_max))
    iters_da = DistArray.from_localarrays(iters_key[0], context=context)
    return iters_da


def numpy_julia_calc(ndarray, c, z_max=10, n_max=100):
    ''' Calculate entirely with NumPy for comparison. '''

    @numpy.vectorize
    def julia_calc(z, c, z_max, n_max):
        ''' Use usual numpy.vectorize to apply on all the complex points. '''
        n = 0
        while abs(z) < z_max and n < n_max:
            z = z * z + c
            n += 1
        return n

    num_iters = julia_calc(ndarray, c, z_max, n_max)
    return num_iters


# Grid parameters
re_ax = (-1.5, 1.5)
im_ax = (-1.5, 1.5)
dimensions = (256, 256)
#dimensions = (500, 500)
#dimensions = (1000, 1000)
# Julia set parameters, changing these is fun.
c = complex(float(sys.argv[1]), float(sys.argv[2]))
z_max = 10
#n_max = 100
n_max = 1000
#n_max = 5000
# Array distribution parameters
#dist = {0: 'c', 1: 'c'}
#dist = {0: 'b', 1: 'b'}


def do_julia_run(context, dist_code, dimensions, c, re_ax, im_ax, plot, verbose):
    # Create dist dictionary.
    dist = {0: dist_code, 1: dist_code}
    # Create a distarray for the points on the complex plane.
    complex_plane = create_complex_plane(context,
                                         dimensions,
                                         dist,
                                         re_ax,
                                         im_ax)
    # Calculate the number of iterations to escape for each point.
    t0 = time()
    num_iters = distributed_julia_calc(complex_plane,
                                       c,
                                       z_max=z_max,
                                       n_max=n_max)
    t1 = time()
    t_distarray = t1 - t0
    # Now try with numpy.
    complex_plane_nd = complex_plane.tondarray()
    t0 = time()
    num_iters_nd = numpy_julia_calc(complex_plane_nd,
                                    c,
                                    z_max=z_max,
                                    n_max=n_max)
    t1 = time()
    t_numpy = t1 - t0
    # Average iteration count.
    avg_iters = num_iters.mean().tondarray()
    if verbose:
        print 'Distribution:', dist
        print 'Num engines:', len(context.targets)
        print 'Average iterations:', avg_iters, 'c:', c
        print 'Elapsed time:', t_distarray
        print 'NumPy elapsed time:', t_numpy
    if plot:
        # Plot the iteration count.
        #image = num_iters.tondarray()
        image = num_iters_nd
        pyplot.matshow(image)
        pyplot.show()
    return avg_iters


if __name__ == '__main__':
    if True:
        dist_code = 'b'
        dist_code = 'c'
        do_julia_run(context, dist_code, dimensions, c, re_ax, im_ax, plot=True, verbose=True)
    else:
        dist_code = 'b'
        dist_code = 'c'
        # Try many values to find something needs lots of iters.
        cxs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        cys = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        for cx in cxs:
            for cy in cys:
                c = complex(cx * 5.0, cy * 5.0)
                avg_iters = do_julia_run(context, dist_code, dimensions, c, re_ax, im_ax, plot=False, verbose=False)
                print 'Average iterations:', avg_iters, 'c:', c
