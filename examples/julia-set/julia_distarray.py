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
from distarray.dist.decorators import local, vectorize


context = Context()
with context.view.sync_imports():
    import numpy


# Make an empty distributed array
def make_empty_da(resolution, dist):
    """Create the arr we will build the fractal with."""

    distribution = Distribution(context, (resolution[0], resolution[1]),
                                dist=dist)
    out = context.empty(distribution, dtype=complex)
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


# This exactly the same function as the one in julia_numpy.py, but here
# we use distarray's vectorize decorator.
@vectorize
def julia_calc(z, c, z_max, n_max):
    n = 0
    fn = lambda z, c: z**2 + c
    while abs(z) < z_max and n < n_max:
        z = fn(z, c)
        n += 1
    return n

# Grid parameteres
re_ax = (-1.5, 1.5)
im_ax = (-1.5, 1.5)
dimensions = (500, 500)
# Julia set parameters, changing these is fun.
c = complex(float(sys.argv[1]), float(sys.argv[2]))
z_max = 10
n_max = 100
# Array distribution parameters
dist = {0: 'c', 1: 'c'}

if __name__ == '__main__':
    # make empty distarray
    out = make_empty_da(dimensions, dist)
    # draw the complex coordinate plane.
    out = draw_coord(out, re_ax, im_ax, dimensions)
    # draw the julia set.
    out = julia_calc(out, c, z_max, n_max)
    # display
    out = numpy.absolute(out.tondarray()).astype(float)
    pyplot.matshow(out)
    pyplot.show()
