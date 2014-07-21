# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Benchmark calculating the Julia set with DistArray for various array
distributions and number of engines.

Usage:
    $ dacluster start -n20
    ...
    $ python bench_dist.py
"""

from __future__ import division

import sys
from timeit import default_timer as clock

import numpy as np
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
# faster than trying to do it by indexing a DistArray.
def draw_coord(arr, re_ax, im_ax, resolution):
    """Draw the complex coordinate plane."""
    re_step = float(re_ax[1] - re_ax[0]) / resolution[0]
    im_step = float(im_ax[1] - im_ax[0]) / resolution[1]
    for i in arr.distribution[0].global_iter:
        for j in arr.distribution[1].global_iter:
            arr.global_index[i, j] = complex(re_ax[0] + re_step*i,
                                             im_ax[0] + im_step*j)
    return arr


# This is exactly the same function as the one in julia_numpy.py, but here
# we use DistArray's vectorize decorator.
def julia(z, c, z_max, n_max):
    n = 0
    fn = lambda z, c: z**2 + c
    while abs(z) < z_max and n < n_max:
        z = fn(z, c)
        n += 1
    return n


def test_distarray(dist, context, resolution, c, re_ax, im_ax, z_max, n_max):
    """Time a function call."""
    local_draw_coord = local(draw_coord)
    vect_julia = vectorize(julia)
    darr = make_empty_da(resolution, dist, context)
    darr = local_draw_coord(darr, re_ax, im_ax, resolution)
    start = clock()
    darr = vect_julia(darr, c, z_max, n_max)
    stop = clock()
    return stop - start


def plot_results(dist_data, dists, engines):
    """Plot the computed timings."""
    from itertools import cycle
    linestyles = ["-", "--", "-.", ":"]
    cycler = cycle(linestyles)
    try:  # nicer plotting defaults
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass

    for data, label in zip(dist_data, dists):
        pyplot.plot(list(engines), data, next(cycler), label=label)

    pyplot.title('Julia Set Benchmark')
    pyplot.xticks(list(engines), list(engines))
    pyplot.xlabel('nengines')
    pyplot.ylabel('1k points / s')
    pyplot.legend(loc='upper right')
    pyplot.savefig("julia_timing.png", dpi=100)
    pyplot.show()


def main():
    # Grid parameters
    re_ax = (-1., 1.)
    im_ax = (-1., 1.)
    resolution = (480, 480)
    # Julia set parameters, changing these is fun.
    c = complex(0., .75)
    z_max = 20
    n_max = 100

    # benchmark parameters
    # number of engines
    #engines = range(1, 20, 1)
    engines = range(1, 3)

    # array distributions
    dists = ['bn', 'cn', 'bc', 'cb', 'bb', 'cc']

    dist_data = [[] for i in range(len(dists))]
    nreps = 3

    for num_engines in engines:
        targets = list(range(num_engines))
        context = Context(targets=targets)
        print("nengines:", num_engines)
        print("dist:", end=" ")
        for i, dist in enumerate(dists):
            print(dist, end=" ")
            sys.stdout.flush()
            times = []
            for j in range(nreps):
                time = test_distarray(dist, context, resolution,
                                      c, re_ax, im_ax, z_max, n_max)
                times.append(time)
            dist_data[i].append(min(times))
        print()

    kpoints = np.array(resolution).prod() / 1000
    data =  kpoints / np.array(dist_data)
    plot_results(data, dists, engines)


if __name__ == '__main__':
    main()
