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
from matplotlib import pyplot

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray
from distarray.dist.decorators import local


context = Context()
with context.view.sync_imports():
    import numpy


# Make an empty distributed array
def make_empty_da(resolution, dist, dtype):
    """Create the arr we will build the fractal with."""

    distribution = Distribution.from_shape(context,
                                           (resolution[0], resolution[1]),
                                           dist=dist)
    out = context.empty(distribution, dtype=dtype)
    return out


# Drawing the coordinate plane directly like this is currently much
# faster than trying to do it by indexing a distarray.
@local
def draw_coord(arr, re_ax, im_ax, resolution):
    """Draw the complex coordinate plane"""
    re_step = float(re_ax[1] - re_ax[0]) / resolution[0]
    im_step = float(im_ax[1] - im_ax[0]) / resolution[1]
    for i in arr.distribution[0].global_iter:
        for j in arr.distribution[1].global_iter:
            arr.global_index[i, j] = complex(re_ax[0] + re_step*i,
                                             im_ax[0] + im_step*j)
    return arr


def local_julia_calc(la, c, z_max, n_max):
    ''' Calculate the number of iterations for the point to escape. '''

    @numpy.vectorize
    def julia_calc(z, c, z_max, n_max):
        ''' Use usual numpy.vectorize to apply on all the complex points. '''
        n = 0
        while abs(z) < z_max and n < n_max:
            z = z*z + c
            n += 1
        return n

    from distarray.local import LocalArray
    a = la.ndarray
    b = julia_calc(a, c, z_max=z_max, n_max=n_max)
    res = LocalArray(la.distribution, buf=b)
    rtn = proxyize(res)
    return rtn


def distributed_julia_calc(distarray, c, z_max=10, n_max=100):
    ''' Calculate the Julia set for an array of points in the complex plane. '''
    context = distarray.context
    iters_key = context.apply(local_julia_calc,
                              (distarray.key, c, z_max, n_max))
    iters_da = DistArray.from_localarrays(iters_key[0], context=context)
    return iters_da


# Grid parameters
re_ax = (-1.5, 1.5)
im_ax = (-1.5, 1.5)
dimensions = (500, 500)
# Julia set parameters, changing these is fun.
c = complex(float(sys.argv[1]), float(sys.argv[2]))
z_max = 10
n_max = 100
# Array distribution parameters
dist = {0: 'c', 1: 'c'}
dist = {0: 'b', 1: 'b'}

if __name__ == '__main__':
    # Create a distarray for the points on the complex plane.
    complex_plane = make_empty_da(dimensions, dist, dtype=complex)
    complex_plane = draw_coord(complex_plane, re_ax, im_ax, dimensions)
    # Calculate the number of iterations to escape.
    num_iters = distributed_julia_calc(complex_plane,
                                       c,
                                       z_max=z_max,
                                       n_max=n_max)
    # Draw the iteration count.
    image = num_iters.tondarray()
    pyplot.matshow(image)
    pyplot.show()
