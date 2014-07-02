# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate some Julia sets using DistArray, and measure the performance.

The Julia set, for a given complex number c, is the set of points z,
such that the repeated iteration z = z**2 + c never escapes to infinity.

This can be plotted by counting how many iterations are required for the
magnitude of z to exceed a cutoff. (For example, if abs(z) > 2, then it
it certain that the point will go off to infinity.)

Depending on the value of c, the Julia set may be connected and contain
a lot of points, or it could be disconnected and contain fewer points.
The points in the set will require the maximum iteration count, so
the connected sets will usually take longer to compute.

Usage:
    $ python julia_example.py

    This will try various parameters, such as the engine count,
    distribution method, resolution, and c value, and print the
    timing results from each run to standard output.

    Or,
    $ python julia_example.py <c real component> <c imaginary component>

    This will use the c value specified on the command line,
    and only vary only the resolution. A plot is shown for each resolution.
"""

from __future__ import print_function

import sys
from time import time
from matplotlib import pyplot

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray
from distarray.dist.decorators import local


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


def distributed_julia_calc(distarray, c, z_max, n_max):
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


def numpy_julia_calc(ndarray, c, z_max, n_max):
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


def do_julia_run(context, dist_code, dimensions, c, re_ax, im_ax, z_max, n_max, plot):
    ''' Do the Julia set calculation and print timing results. '''
    # Create dist dictionary.
    dist = {0: dist_code, 1: dist_code}
    num_engines = len(context.targets)
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
    avg_iters = float(num_iters.mean().tondarray())
    # Print results.
    t_ratio = t_numpy / t_distarray
    result = '%s, %r, %r, %r, %r, %r, %r, %r' % (
                 dist_code, num_engines, dimensions[0],
                 t_distarray, t_numpy, t_ratio,
                 avg_iters, str(c))
    print(result)
    if plot:
        # Plot the iteration count.
        image = num_iters.tondarray()
        pyplot.matshow(image)
        pyplot.show()
    return avg_iters


def do_julia_runs(context,
                  repeat_count,
                  engine_count_list,
                  dist_code_list,
                  resolution_list,
                  c_list,
                  re_ax, im_ax, z_max, n_max, plot):
    ''' Perform a series of Julia set calculations, and print the results. '''
    # Check that we have enough engines available.
    max_engine_count = max(engine_count_list)
    num_engines = len(context.targets)
    print('%d Engines available' % (num_engines))
    if max_engine_count > num_engines:
        msg = 'Require %d engines, but only %d are available.' % (
            max_engine_count, num_engines)
        raise ValueError(msg)
    # Loop over everything and time the calculations.
    print('Dist, Engines, Resolution, t_DistArray, t_NumPy, t_Ratio, Iters, c')
    for i in range(repeat_count):
        for engine_count in engine_count_list:
            context_use = Context(targets=range(engine_count))
            for dist_code in dist_code_list:
                for resolution in resolution_list:
                    dimensions = (resolution, resolution)
                    for c in c_list:
                        do_julia_run(context_use,
                                     dist_code,
                                     dimensions,
                                     c,
                                     re_ax, im_ax,
                                     z_max, n_max, plot)


if __name__ == '__main__':

    context = Context()
    with context.view.sync_imports():
        import numpy

    # Fixed parameters:

    # Region of the complex plane.
    re_ax = (-1.5, 1.5)
    im_ax = (-1.5, 1.5)

    # Size of number that we consider as off to infinity.
    # I think that 2 is sufficient to be sure that the point will escape.
    z_max = 2

    # Maximum iteration counts. Points in the set will hit this limit,
    # so increasing this has a large effect on the run-time.
    n_max = 100
    #n_max = 1000
    #n_max = 5000
    #n_max = 10000

    # Lists of parameters:

    # Distribution types to use.
    dist_code_list = ['b', 'c']
    #dist_code_list = ['b']

    # Constants to use.
    c_list = [complex(-0.045, 0.45)]      # This Julia set has many points inside, needing all iterations.
    # Or try lots of values over a grid. 
    #cx_list = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    #cy_list = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    #c_list = [complex(cx, cy) for cx in cx_list for cy in cy_list]

    # Number of engines to use.
    engine_count_list = [4]
    #engine_count_list = [2, 3, 4]

    # Resolution of Julia set.
    #resolution_list = [256]
    resolution_list = [16, 32, 64, 128, 256, 512]

    # Number of cycles to repeat everything.
    #repeat_count = 1
    repeat_count = 2

    # If we got command line parameters for c, then use these,
    # and only loop over the resolutions, making plots.
    # This lets you interactively try values.
    # Otherwise, we loop over all parameters.
    if len(sys.argv) == 3:
        # Get constant from command line instead.
        c = complex(float(sys.argv[1]), float(sys.argv[2]))
        c_list = [c]
        do_julia_runs(context,
                      repeat_count=1,
                      engine_count_list=[len(context.targets)],
                      dist_code_list=['b'],
                      resolution_list=resolution_list,
                      c_list=c_list,
                      re_ax=re_ax, im_ax=im_ax,
                      z_max=z_max, n_max=n_max, plot=True)
    else:
        # Normal case, loop over all parameter lists.
        do_julia_runs(context,
                      repeat_count,
                      engine_count_list,
                      dist_code_list,
                      resolution_list,
                      c_list,
                      re_ax, im_ax, z_max, n_max, plot=False)
