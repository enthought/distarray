# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate some Julia sets using DistArray and measure the performance.

The Julia set, for a given complex number c, is the set of points z
such that the repeated iteration z = z**2 + c never escapes to infinity.

This can be plotted by counting how many iterations are required for the
magnitude of z to exceed a cutoff. (For example, if abs(z) > 2, then it
it certain that the point will go off to infinity.)

Depending on the value of c, the Julia set may be connected and contain
a lot of points, or it could be disconnected and contain fewer points.
The points in the set will require the maximum iteration count, so
the connected sets will usually take longer to compute.
"""

from __future__ import print_function

import argparse
from time import time
from contextlib import closing

import numpy
from matplotlib import pyplot

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray


def numpy_julia_calc(z, c, z_max, n_max):
    """Calculate entirely with NumPy for comparison.

    Parameters
    ----------
    z : NumPy array
        array of complex values whose iterations we will count.
    c : complex
        Complex number to add at each iteration.
    z_max : float
        Magnitude of complex value that we assume goes to infinity.
    n_max : int
        Maximum number of iterations.
    """
    counts = numpy.zeros_like(z, dtype=numpy.int32)
    hits = numpy.zeros_like(z, dtype=numpy.bool)
    mask = numpy.zeros_like(z, dtype=numpy.bool)
    n = 0

    while not numpy.all(hits) and n < n_max:
        z = z * z + c
        mask = (abs(z) > z_max) & (~hits)
        counts[mask] = n
        hits |= mask
        z[hits] = 0
        n += 1
    counts[~hits] = n_max
    return counts


def create_complex_plane(context, resolution, dist, re_ax, im_ax):
    """Create a DistArray containing points on the complex plane.

    Parameters
    ----------
    context : DistArray Context
    resolution : 2-tuple
        The number of points along Re and Im axes.
    dist : 2-element sequence or dict
        dist_type for of the DistArray Distribution.
    re_ax : 2-tuple
        The (lower, upper) range of the Re axis.
    im_ax : 2-tuple
        The (lower, upper) range of the Im axis.
    """

    def fill_complex_plane(arr, re_ax, im_ax, resolution):
        """Fill in points on the complex coordinate plane."""
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
    distribution = Distribution(context, (resolution[0], resolution[1]),
                                dist=dist)
    complex_plane = context.empty(distribution, dtype=complex)
    context.apply(fill_complex_plane,
                  (complex_plane.key, re_ax, im_ax, resolution))
    return complex_plane


def local_julia_calc(la, c, z_max, n_max):
    """Calculate the number of iterations for the point to escape.

    Parameters
    ----------
    la : LocalArray
        LocalArray of complex values whose iterations we will count.
    c : complex
        Complex number to add at each iteration.
    z_max : float
        Magnitude of complex value that we assume goes to infinity.
    n_max : int
        Maximum number of iterations.
    """
    from distarray.local import LocalArray
    counts = numpy_julia_calc(la.ndarray, c, z_max, n_max)
    res = LocalArray(la.distribution, buf=counts)
    return proxyize(res)  # noqa


def distributed_julia_calc(distarray, c, z_max, n_max):
    """Calculate the Julia set for an array of points in the complex plane.

    Parameters
    ----------
    distarray : DistArray
        DistArray of complex values whose iterations we will count.
    c : complex
        Complex number to add at each iteration.
    z_max : float
        Magnitude of complex value that we assume goes to infinity.
    n_max : int
        Maximum number of iterations.
    """
    context = distarray.context
    iters_key = context.apply(local_julia_calc,
                              (distarray.key, c, z_max, n_max))
    iters_da = DistArray.from_localarrays(iters_key[0], context=context,
                                          dtype=numpy.int32)
    return iters_da


def do_julia_run(context, dist, dimensions, c, re_ax, im_ax, z_max, n_max,
                 plot, benchmark_numpy=False):
    """Do the Julia set calculation and print timing results.

    Parameters
    ----------
    context : DistArray Context
    dist : 2-element sequence
        Distribution type to test.  Example: 'bc'
    dimensions : 2-tuple of int
        Dimensions of complex plane to use.
    c_list : complex
        Constant to use to compute Julia set.  Example: complex(-0.045, 0.45)
    re_ax : 2-tuple of float
        Min and max for real axis.
    im_ax : 2-tuple of float
        Min and max for imaginary axis.
    z_max : float
        Size of number that we consider as going off to infinity.  I think that
        2.0 is sufficient to be sure that the point will escape.
    n_max : int
        Maximum iteration counts. Points in the set will hit this limit, so
        increasing this has a large effect on the run-time.
    plot : bool
        Make plots of the computed Julia sets.
    benchmark_numpy : bool
        Compute with numpy instead of DistArray?
    """
    num_engines = len(context.targets)
    # Calculate the number of iterations to escape for each point.
    if benchmark_numpy:
        complex_plane = create_complex_plane(context, dimensions, 'bb',
                                             re_ax, im_ax)
        complex_plane_nd = complex_plane.tondarray()
        t0 = time()
        num_iters = numpy_julia_calc(complex_plane_nd, c,
                                     z_max=z_max, n_max=n_max)
        t1 = time()
        iters_list = [num_iters.sum()]
    else:
        complex_plane = create_complex_plane(context, dimensions, dist,
                                             re_ax, im_ax)
        t0 = time()
        num_iters = distributed_julia_calc(complex_plane, c,
                                           z_max=z_max, n_max=n_max)
        t1 = time()
        # Iteration count.
        def local_sum(la):
            return la.ndarray.sum()
        iters_list = context.apply(local_sum, (num_iters.key,))
    t_distarray = t1 - t0

    # Print results.
    dist_text = dist if dist=='numpy' else '-'.join(dist)

    fmt = '%s, %s, %r, %r, %r, %r, %r'
    result = fmt % (time(), dist_text, dimensions[0], c, num_engines,
                    t_distarray, str(iters_list))
    print(result)
    if plot:
        # Plot the iteration count.
        image = num_iters.tondarray()
        pyplot.matshow(image)
        pyplot.show()
    return sum(iters_list)


def do_julia_runs(repeat_count, engine_count_list, dist_list, resolution_list,
                  c_list, re_ax, im_ax, z_max, n_max, plot):
    """Perform a series of Julia set calculations, and print the results.

    Loop over all parameter lists.

    Parameters
    ----------
    repeat_count : int
        Number of times to repeat each unique parameter set.  Later we can take
        the average or minimum of these values to reduce noise in the output.
    engine_count_list : list of int
        List of numbers of engines to test.  Example: list(range(1, 5))
    dist_list : list of 2-element sequences
        List of distribution types to test.  Example: ['bn', 'cn', 'bb', 'cc']
    resolution_list = list of int
        List of resolutions of Julia set to test.
    c_list : list of complex
        Constants to use to compute Julia set.
        Example: [complex(-0.045, 0.45)]
    re_ax : 2-tuple of float
        Min and max for real axis.
    im_ax : 2-tuple of float
        Min and max for imaginary axis.
    z_max : float
        Size of number that we consider as going off to infinity.  I think that
        2.0 is sufficient to be sure that the point will escape.
    n_max : int
        Maximum iteration counts. Points in the set will hit this limit, so
        increasing this has a large effect on the run-time.
    plot : bool
        Make plots of the computed Julia sets.
    """
    max_engine_count = max(engine_count_list)
    with closing(Context()) as context:
        # Check that we have enough engines available.
        num_engines = len(context.targets)

    title = 'Julia Set Performance'
    print(title)
    print("Benchmark started: %s" % time())
    fmt = ' ,'.join(('num_engines=%d', 'z_max=%r', 'n_max=%r', 're_ax=%r',
                     'im_ax=%r', 'repeat_count=%r'))
    msg = fmt % (num_engines, z_max, n_max, re_ax, im_ax, repeat_count)
    print(msg)
    if max_engine_count > num_engines:
        msg = 'Require %d engines, but only %d are available.' % (
            max_engine_count, num_engines)
        raise ValueError(msg)

    # Loop over everything and time the calculations.
    hdr = ', '.join(('Timestamp', 'Dist', 'Engines', 'Resolution', 'c',
                     't_DistArray', 'Iters'))
    print(hdr)
    for i in range(repeat_count):
        for resolution in resolution_list:
            dimensions = (resolution, resolution)
            # numpy julia run
            for c in c_list:
                with closing(Context(targets=[0])) as context:
                    do_julia_run(context, 'numpy', dimensions, c, re_ax, im_ax,
                                 z_max, n_max, plot, benchmark_numpy=True)
                for engine_count in engine_count_list:
                    for dist in dist_list:
                        targets = list(range(engine_count))
                        with closing(Context(targets=targets)) as context:
                            context.register(numpy_julia_calc)
                            do_julia_run(context, dist, dimensions, c, re_ax,
                                         im_ax, z_max, n_max, plot,
                                         benchmark_numpy=False)


def cli(cmd):
    """
    Process command line arguments, set default params, and do_julia_runs.

    Parameters
    ----------
    cmd : list of str
        sys.argv
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resolution_list',  metavar='N', type=int, nargs='+',
                        help="resolutions of the Julia set to benchmark (NxN)")
    parser.add_argument("-r", "--repeat", type=int, dest='repeat_count',
                        default=3,
                        help=("number of repetitions of each unique parameter "
                              "set, default: 3"))
    args = parser.parse_args()

    ## Default parameters
    with closing(Context()) as context:
        # use all available targets
        engine_count_list = list(range(1, len(context.targets) + 1))
    dist_list = ['bn', 'cn', 'bb', 'cc']
    c_list = [complex(-0.045, 0.45)]  # This Julia set has many points inside
                                      # needing all iterations.
    re_ax = (-1.5, 1.5)
    im_ax = (-1.5, 1.5)
    z_max = 2.0
    n_max = 100

    do_julia_runs(args.repeat_count, engine_count_list, dist_list,
                  args.resolution_list, c_list, re_ax, im_ax, z_max, n_max,
                  plot=False)


if __name__ == '__main__':
    import sys
    cli(sys.argv)