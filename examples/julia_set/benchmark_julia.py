# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate some Julia sets using DistArray and measure the performance.

The Julia set, for a given complex number $c$, is the set of points $z$
such that $|z_{i}|$ remains bounded where $z_{i+1} = z_{i}^2 + c$.

This can be plotted by counting how many iterations are required for $|z_{i}|$
to exceed a cutoff.

Depending on the value of $c$, the Julia set may be connected and contain
a lot of points, or it could be disconnected and contain fewer points.
The points in the set will require the maximum iteration count, so
the connected sets will usually take longer to compute.
"""

from __future__ import print_function

import argparse
import json
from time import time
from contextlib import closing
from math import sqrt, floor

import numpy

from distarray.globalapi import Context, Distribution
from distarray.globalapi.distarray import DistArray


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
    z = numpy.asarray(z)
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


def fancy_numpy_julia_calc(z, c, z_max, n_max):
    """Calculate entirely with NumPy, using fancy indexing.

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
    z = numpy.asarray(z)
    counts = numpy.zeros_like(z, dtype=numpy.int32)
    hits = numpy.zeros_like(z, dtype=numpy.bool)
    mask = numpy.zeros_like(z, dtype=numpy.bool)
    n = 0

    while not numpy.all(hits) and n < n_max:
        z[~hits] = z[~hits] * z[~hits] + c
        mask = (abs(z) > z_max) & (~hits)
        counts[mask] = n
        hits |= mask
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
    import numpy as np
    from kernel import fill_complex_plane

    # Create an empty distributed array.
    distribution = Distribution(context, (resolution[0], resolution[1]),
                                dist=dist)
    complex_plane = context.empty(distribution, dtype=np.complex64)
    context.apply(fill_complex_plane,
                  (complex_plane.key, re_ax, im_ax, resolution))
    return complex_plane


def local_julia_calc(la, c, z_max, n_max, kernel):
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
    kernel : function
        Kernel to use for computation of the Julia set.  Options are 'fancy',
        'numpy', or 'cython'.
    """
    from distarray.localapi import LocalArray
    counts = kernel(la, c, z_max, n_max)
    res = LocalArray(la.distribution, buf=counts)
    return proxyize(res)  # noqa


def distributed_julia_calc(distarray, c, z_max, n_max,
                           kernel=fancy_numpy_julia_calc):
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
    kernel: function
        Kernel to use for computation of the Julia set.  Options are 'fancy',
        'numpy', or 'cython'.
    """
    context = distarray.context
    iters_key = context.apply(local_julia_calc,
                              (distarray.key, c, z_max, n_max),
                              {'kernel': kernel})
    iters_da = DistArray.from_localarrays(iters_key[0], context=context,
                                          dtype=numpy.int32)
    return iters_da


def do_julia_run(context, dist, dimensions, c, complex_plane, z_max, n_max,
                 benchmark_numpy=False, kernel=fancy_numpy_julia_calc):
    """Do the Julia set calculation and print timing results.

    Parameters
    ----------
    context : DistArray Context
    dist : 2-element sequence
        Distribution type to test.  Example: 'bc'
    dimensions : 2-tuple of int
        Dimensions of complex plane to use.
    c : complex
        Constant to use to compute Julia set.  Example: complex(-0.045, 0.45)
    complex_plane: DistArray
        DistArray of the initial complex plane for iteration.
    z_max : float
        Size of number that we consider as going off to infinity.  I think that
        2.0 is sufficient to be sure that the point will escape.
    n_max : int
        Maximum iteration counts. Points in the set will hit this limit, so
        increasing this has a large effect on the run-time.
    benchmark_numpy : bool
        Compute with numpy instead of DistArray?
    kernel : function
        Kernel to use for computation of the Julia set.  Options are 'fancy',
        'numpy', or 'cython'.
    """
    num_engines = len(context.targets)
    # Calculate the number of iterations to escape for each point.
    if benchmark_numpy:
        complex_plane_nd = complex_plane.tondarray()
        t0 = time()
        num_iters = kernel(complex_plane_nd, c, z_max=z_max, n_max=n_max)
        t1 = time()
        iters_list = [numpy.asscalar(numpy.asarray(num_iters).sum())]
    else:
        t0 = time()
        num_iters = distributed_julia_calc(complex_plane, c,
                                           z_max=z_max, n_max=n_max,
                                           kernel=kernel)
        t1 = time()

        # Iteration count.
        def local_sum(la):
            return numpy.asscalar(la.ndarray.sum())
        iters_list = context.apply(local_sum, (num_iters.key,))

    # Print results.
    dist_text = dist if dist == 'numpy' else '-'.join(dist)

    return (t0, t1, dist_text, dimensions[0], str(c), num_engines, iters_list)


def do_julia_runs(repeat_count, engine_count_list, dist_list, resolution_list,
                  c_list, re_ax, im_ax, z_max, n_max, output_filename,
                  kernel=fancy_numpy_julia_calc, scaling="strong"):
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
    output_filename : str
    kernel : function
        Kernel to use for computation of the Julia set.  Options are 'fancy',
        'numpy', or 'cython'.
    scaling: str, either "strong" or "weak"
    """
    max_engine_count = max(engine_count_list)
    with closing(Context()) as context:
        # Check that we have enough engines available.
        num_engines = len(context.targets)
    if max_engine_count > num_engines:
        msg = 'Require %d engines, but only %d are available.' % (
            max_engine_count, num_engines)
        raise ValueError(msg)

    # Loop over everything and time the calculations.
    results = []
    hdr = (('Start', 'End', 'Dist', 'Resolution', 'c', 'Engines', 'Iters'))
    print("(n/n_runs: time)", hdr)
    # progress stats
    n_regular_runs = repeat_count * (len(resolution_list) * len(c_list) *
                                     len(engine_count_list) * len(dist_list))
    n_numpy_runs = repeat_count * (len(resolution_list) * len(c_list))
    n_runs = n_regular_runs + n_numpy_runs
    prog_fmt = "({:d}/{:d}: {:0.3f}s)"
    n = 0
    for i in range(repeat_count):
        for resolution in resolution_list:
            dimensions = (resolution, resolution)
            for c in c_list:
                with closing(Context(targets=[0])) as context:
                    # numpy julia run
                    complex_plane = create_complex_plane(context, dimensions,
                                                         'bn', re_ax, im_ax)
                    result = do_julia_run(context, 'numpy', dimensions, c,
                                          complex_plane, z_max, n_max,
                                          benchmark_numpy=True, kernel=kernel)
                    results.append({h: r for h, r in zip(hdr, result)})
                    n += 1
                    print(prog_fmt.format(n, n_runs, result[1] - result[0]), result)
                for engine_count in engine_count_list:
                    if scaling == "weak":
                        factor = sqrt(engine_count)
                        dimensions = (int(floor(resolution * factor)),) * 2
                    for dist in dist_list:
                        targets = list(range(engine_count))
                        with closing(Context(targets=targets)) as context:
                            context.register(kernel)
                            complex_plane = create_complex_plane(context,
                                                                 dimensions,
                                                                 dist, re_ax,
                                                                 im_ax)
                            result = do_julia_run(context, dist, dimensions, c,
                                                  complex_plane, z_max, n_max,
                                                  benchmark_numpy=False,
                                                  kernel=kernel)
                            results.append({h: r for h, r in zip(hdr, result)})
                            n += 1
                            print(prog_fmt.format(n, n_runs, result[1] - result[0]), result)
                            with open(output_filename, 'wt') as fp:
                                json.dump(results, fp, sort_keys=True,
                                          indent=4, separators=(',', ': '))
    return results


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
    parser.add_argument("-o", "--output-filename", type=str,
                        dest='output_filename', default='out.json',
                        help=("filename to write the json data to."))
    parser.add_argument("-k", "--kernel", type=str, default='fancy',
                        choices=("fancy", "numpy", "cython"),
                        help=("kernel to use for computation.  "
                              "Options are 'fancy', 'numpy', or 'cython'."))
    parser.add_argument("-s", "--scaling", type=str, default="strong",
                        choices=("strong", "weak"),
                        help=("Kind of scaling test.  Options are 'strong' or 'weak'"))
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

    fn_from_kernel = {'fancy': fancy_numpy_julia_calc,
                      'numpy': numpy_julia_calc}

    if args.kernel == 'cython':
        from kernel import cython_julia_calc
        fn_from_kernel['cython'] = cython_julia_calc

    results = do_julia_runs(args.repeat_count, engine_count_list, dist_list,
                            args.resolution_list, c_list, re_ax, im_ax, z_max,
                            n_max, output_filename=args.output_filename,
                            kernel=fn_from_kernel[args.kernel],
                            scaling=args.scaling)


if __name__ == '__main__':
    import sys
    cli(sys.argv)
