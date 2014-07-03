# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Benchmark calculating the Julia set with Distarray for various array
distributions and number of engines. Usage:
    $ ipcluster start --n=20 --engines=MPI
    ...
    $ python bench_dist.py
"""

from timeit import default_timer as clock

from IPython.parallel import Client
from matplotlib import pyplot

from distarray.dist import Context, Distribution
from distarray.dist.decorators import local, vectorize


# Make an empty distributed array
def make_empty_da(resolution, dist, context):
    """Create the arr we will build the fractal with."""
    distribution = Distribution(context, (resolution[0], resolution[1]),
                                dist=dist)
    out = context.empty(distribution, dtype=complex)
    return out


# Drawing the coordinate plane directly like this is currently much
# faster than trying to do it by indexing a distarray.
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
def julia(z, c, z_max, n_max):
    n = 0
    fn = lambda z, c: z**2 + c
    while abs(z) < z_max and n < n_max:
        z = fn(z, c)
        n += 1
    return n


def test_distarray(dist, context):
    global draw_coord
    global julia
    local_draw_coord = local(draw_coord)
    vect_julia = vectorize(julia)
    darr = make_empty_da(resolution, dist, context)
    darr = local_draw_coord(darr, re_ax, im_ax, resolution)
    start = clock()
    darr = vect_julia(darr, c, z_max, n_max)
    stop = clock()
    return stop - start


# Grid parameteres
re_ax = (-1., 1.)
im_ax = (-1., 1.)
resolution = (480, 480)
# Julia set parameters, changing these is fun.
c = complex(0., .75)
z_max = 20
n_max = 100

# benchmark parameters
# number of engines
engines = range(4, 21, 2)

# array distributions
dists = [{0: 'c', 1: 'b'}, {0: 'c', 1: 'c'}, {0: 'b'}, {0: 'c'},
         {0: 'b', 1: 'b'}, {0: 'b', 1: 'c'}]

dist_data = [[] for i in range(len(dists))]
engine_data = []
client = Client()

for num_engines in engines:
    targets = list(range(num_engines))
    context = Context(client, targets=targets)
    print(num_engines)
    for i, dist in enumerate(dists):
        print(dist)
        time = test_distarray(dist, context)
        dist_data[i].append(time)

for i, data in enumerate(dist_data):
    pyplot.plot(list(engines), data, label=dists[i].__repr__(), lw=2)

pyplot.title('Julia set benchmark - array distribution type vs number of '
             'engines')
pyplot.xticks(list(engines), list(engines))
pyplot.xlabel('number of engines')
pyplot.ylabel('time (s)')
pyplot.legend(loc='upper right')
pyplot.show()
