# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Load the seismic volume, created with create_volume.py, and do stuff with it.

Usage:
    $ python load_volume.py --file <filename> --key <keyname> --hdf5 --dnpy

The --file argument specifies the input filename, default 'seismic.hdf5'.
The --key argument specifies the name of the data block in the HDF5 file, default 'seismic'.
The --hdf5 argument specifies to read from an HDF5 file, default True.
The --dnpy argument specifies to read from .dnpy files, default False.
Only one of --hdf5 or --dnpy can be specified.

The volume is loaded in parallel as a DistArray.
It can be read from either a single HDF5 file or a set of .dnpy files.
Each trace (a z-slice constant in x and y) has statistics calculated for it.
Next we apply a couple of filters to each trace.
Next we extract some slices from the volume and create plots for them.
We also extract slices from one of the filtered results.
Finally, we save the 3-point-average filtered volume, both as an HDF5 file,
and as a set of .dnpy files.
"""

from __future__ import print_function

import argparse
import os.path

import numpy
import h5py
from matplotlib import pyplot

from distarray.globalapi import Context, Distribution
from distarray.globalapi.distarray import DistArray


# Load/save of seismic data from HDF5 for .dnpy files.

def get_hdf5_dataset_shape(pathname, key):
    ''' Get the shape of the array in the HDF5 file. '''
    with h5py.File(pathname, 'r') as f:
        dataset = f[key]
        shape = dataset.shape
    return shape


def load_hdf5_distarray(context, filename, key, dist):
    ''' Create a distarray from the specified section of the HDF5 file. '''
    # Filename for load_hdf5() needs the full path.
    pathname = os.path.abspath(filename)
    # Get array shape.
    print('Getting array shape...')
    array_shape = get_hdf5_dataset_shape(pathname, key)
    # Create distribution.
    print('Creating distribution...')
    distribution = Distribution(context, array_shape, dist=dist)
    # Load HDF5 file into DistArray.
    print('Loading HDF5 file...')
    distarray = context.load_hdf5(filename=pathname,
                                  distribution=distribution,
                                  key=key)
    print('Loaded.')
    return distarray


def save_hdf5_distarray(filename, key, distarray):
    ''' Write a distarray to an HDF5 file. '''
    pathname = os.path.abspath(filename)
    context = distarray.context
    context.save_hdf5(pathname, distarray, key=key, mode='w')


def load_dnpy_distarray(context, filename):
    ''' Create a distarray from the .dnpy files. '''
    # Remove extension from filename.
    filename = os.path.splitext(filename)[0]
    pathname = os.path.abspath(filename)
    print('Loading .dnpy files...')
    distarray = context.load_dnpy(pathname)
    print('Loaded.')
    return distarray


def save_dnpy_distarray(filename, distarray):
    ''' Write a distarray to .dnpy files. '''
    pathname = os.path.abspath(filename)
    context = distarray.context
    context.save_dnpy(pathname, distarray)


# Print some stuff about the array.
# (Mostly only practical for small ones.)

def dump_distarray_info(da):
    ''' Print some stuff about the array. '''
    print('Local Shapes:')
    localshapes = da.localshapes()
    print(localshapes)
    print('Arrays Per Process:')
    ndarrays = da.get_ndarrays()
    for i, ndarray in enumerate(ndarrays):
        print('Process:', i)
        print(ndarray)
    print('Full Array:')
    ndarray = da.toarray()
    print(ndarray)


# Calculate statistics on each trace [z slice].
# We get the same results when calculating via DistArray,
# and via NumPy on the full array.

def calc_distributed_stats(distarray):
    '''Calculate some statistics on each trace [z slices] in the distarray.'''
    trace_min = distarray.min(axis=2)
    trace_max = distarray.max(axis=2)
    trace_mean = distarray.mean(axis=2)
    trace_var = distarray.var(axis=2)
    trace_std = distarray.std(axis=2)
    # Collect into dict.
    distarray_stats = {
        'min': trace_min,
        'max': trace_max,
        'mean': trace_mean,
        'var': trace_var,
        'std': trace_std,
    }
    return distarray_stats


def calc_undistributed_stats(ndarray):
    ''' Calculate some statistics on each trace [z slices] in the ndarray. '''
    trace_min = ndarray.min(axis=2)
    trace_max = ndarray.max(axis=2)
    trace_mean = ndarray.mean(axis=2)
    trace_var = ndarray.var(axis=2)
    trace_std = ndarray.std(axis=2)
    # Collect into dict.
    undistributed_stats = {
        'min': trace_min,
        'max': trace_max,
        'mean': trace_mean,
        'var': trace_var,
        'std': trace_std,
    }
    return undistributed_stats


def compare_stats(distarray_stats, numpy_stats, verbose=False):
    ''' Compare the statistics made on DistArray vs NumPy array. '''

    stat_keys = ['min', 'max', 'mean', 'var', 'std']

    def print_stats(title, stat_dict):
        ''' Print out the statistics. '''
        print(title)
        for stat in stat_keys:
            print(stat)
            print(stat_dict[stat])

    # Convert DistArray to NumPy array to allow comparison.
    for stat in stat_keys:
        distarray_stats[stat] = distarray_stats[stat].toarray()

    # The difference is ideally zero.
    if verbose:
        print('Calculating difference...')
    difference_stats = {}
    for stat in stat_keys:
        diff = distarray_stats[stat] - numpy_stats[stat]
        difference_stats[stat] = diff

    # Print statistics.
    if verbose:
        print_stats('Distributed stats:', distarray_stats)
        print_stats('Undistributed stats:', numpy_stats)
        print_stats('Difference:', difference_stats)

    # Test that results are effectively the same with allclose().
    for stat in stat_keys:
        distributed_stat = distarray_stats[stat]
        undistributed_stat = numpy_stats[stat]
        # For 512x512x1024, single precision, var has errors which exceed
        # the default allclose() bounds, many about 0.004.
        is_close = numpy.allclose(distributed_stat, undistributed_stat)
        print(stat, 'is_close:', is_close)


def analyze_statistics(distarray, compare=True, verbose=False):
    ''' Calculate statistics on each trace, via DistArray and NumPy.

    The results should match within numerical precision.
    '''
    if verbose:
        print('Calculating statistics...')
    # Using DistArray methods.
    if verbose:
        print('DistArray statistics...')
    distributed_stats = calc_distributed_stats(distarray)
    if compare:
        # Convert to NumPy array and use NumPy methods.
        if verbose:
            print('NumPy array statistics...')
        ndarray = distarray.tondarray()
        if verbose:
            print('Converted to ndarray...')
        undistributed_stats = calc_undistributed_stats(ndarray)
        # Compare.
        compare_stats(distributed_stats, undistributed_stats, verbose=verbose)


# 3-point averaging filter. This uses a 2-point average on the ends.

def filter_avg3(a):
    ''' Filter a numpy array via 3-point average over z axis.
    A 2-point average is used at the ends. '''
    from numpy import empty_like
    b = empty_like(a)
    b[:, :, 0] = (a[:, :, 0] + a[:, :, 1]) / 2.0
    b[:, :, 1:-1] = (a[:, :, :-2] + a[:, :, 1:-1] + a[:, :, 2:]) / 3.0
    b[:, :, -1] = (a[:, :, -2] + a[:, :, -1]) / 2.0
    return b


def local_filter_avg3(la):
    ''' Filter a local array via 3-point average over z axis. '''

    def filter_avg3(a):
        ''' Filter a local array via 3-point average over z axis.
        A 2-point average is used at the ends. '''
        from numpy import empty_like
        b = empty_like(a)
        b[:, :, 0] = (a[:, :, 0] + a[:, :, 1]) / 2.0
        b[:, :, 1:-1] = (a[:, :, :-2] + a[:, :, 1:-1] + a[:, :, 2:]) / 3.0
        b[:, :, -1] = (a[:, :, -2] + a[:, :, -1]) / 2.0
        return b

    from distarray.localapi import LocalArray
    a = la.ndarray
    b = filter_avg3(a)
    res = LocalArray(la.distribution, buf=b)
    rtn = proxyize(res)
    return rtn


# 3-point maximum window filter.

def filter_max3(a):
    ''' Filter a numpy array via a 3-element window maximum. '''
    from numpy import empty_like
    b = empty_like(a)
    shape = a.shape
    for k in xrange(shape[2]):
        k0 = max(k - 1, 0)
        k1 = min(k + 1, shape[2] - 1)
        b[:, :, k] = a[:, :, k0:k1 + 1].max(axis=2)
    return b


def local_filter_max3(la):
    ''' Filter a local array via 3-point average over z axis. '''

    def filter_max3(a):
        ''' Filter a numpy array via a 3-element window maximum. '''
        from numpy import empty_like
        b = empty_like(a)
        shape = a.shape
        for k in xrange(shape[2]):
            k0 = max(k - 1, 0)
            k1 = min(k + 1, shape[2] - 1)
            b[:, :, k] = a[:, :, k0:k1 + 1].max(axis=2)
        return b

    from distarray.localapi import LocalArray
    a = la.ndarray
    b = filter_max3(a)
    res = LocalArray(la.distribution, buf=b)
    rtn = proxyize(res)
    return rtn


def distributed_filter(distarray, local_filter):
    ''' Filter a DistArray, returning a new DistArray. '''
    context = distarray.context
    filtered_key = context.apply(local_filter, (distarray.key,))
    filtered_da = DistArray.from_localarrays(filtered_key[0], context=context)
    return filtered_da


def undistributed_filter(ndarray, numpy_filter):
    ''' Filter a NumPy array, returning a new NumPy array. '''
    filtered_nd = numpy_filter(ndarray)
    return filtered_nd


def analyze_filter(da, local_filter, numpy_filter, compare, verbose):
    ''' Apply the filter both via DistArray methods and via NumPy methods. '''
    # Via DistArray.
    result_distarray = distributed_filter(da, local_filter)
    if verbose:
        # Print results of averaging.
        print('Original:')
        print(da.toarray())
        print('Filtered:')
        print(result_distarray.toarray())
    if compare:
        # Filter via NumPy array.
        ndarray = da.toarray()
        result_numpy = undistributed_filter(ndarray, numpy_filter)
        # Difference between DistArray and NumPy results.
        distributed_filtered = result_distarray.toarray()
        undistributed_filtered = result_numpy
        diff = distributed_filtered - undistributed_filtered
        print('Difference of DistArray - NumPy filters:')
        if verbose:
            print(diff)
        is_close = numpy.allclose(distributed_filtered, undistributed_filtered)
        print('is_close:', is_close)
    # Return the filtered distarray.
    return result_distarray


# Slicing examples.

def plot_slice(distarray_slice, filename, title, x_label, y_label):
    ''' Create an array plot of the 2D slice. '''
    print('Visualizing slice', title)
    num_dim = len(distarray_slice.shape)
    if (num_dim != 2):
        raise ValueError('Slice must be 2D for plotting.')
    # Convert to ndarray for plotting.
    slice_nd = distarray_slice.toarray()
    # Transpose to make depth look like depth.
    slice_nd = slice_nd.transpose()
    # Plot.
    cmap = 'RdBu'
    pyplot.matshow(slice_nd, cmap=cmap)
    if False:
        # This looks bad without more effort.
        pyplot.colorbar()
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.savefig(filename, dpi=100)


def slice_volume(distarray, base_filename='slice_'):
    ''' Slice the volume three different ways and plot result. '''
    shape = distarray.shape
    # Choose (arbitrary) indices for slicing.
    i_index = shape[0] // 4
    j_index = (2 * shape[1]) // 3
    k_index = shape[2] // 2
    # Slice and visualize result.
    print('Taking I-Slice...')
    i_slice = distarray[i_index, :, :]
    filename = base_filename + 'i_plot.png'
    title = 'Vertical Slice [i = %d]' % (i_index)
    plot_slice(i_slice, filename, title, 'j', 'depth')
    print('Taking J-Slice...')
    filename = base_filename + 'j_plot.png'
    title = 'Vertical Slice [j = %d]' % (j_index)
    j_slice = distarray[:, j_index, :]
    plot_slice(j_slice, filename, title, 'i', 'depth')
    print('Taking K-Slice...')
    filename = base_filename + 'k_plot.png'
    title = 'Horizontal Slice [k = %d]' % (k_index)
    k_slice = distarray[:, :, k_index]
    plot_slice(k_slice, filename, title, 'i', 'j')


# Main processing functions.

def load_seismic_volume(filename, key, dist, use_hdf5):
    ''' Load the seismic volume, from HDF5 or .dnpy files. '''
    # Create context.
    context = Context()
    if use_hdf5:
        print('Loading from .hdf5 file...')
        da = load_hdf5_distarray(context, filename, key, dist)
    else:
        print('Loading from .dnpy files...')
        da = load_dnpy_distarray(context, filename)
    # Print some stuff about the array.
    if False:
        dump_distarray_info(da)
    return da


def process_seismic_volume(da, key, compare=True, verbose=False):
    ''' Do some processsing with the seismic volume as a DistArray.

    Also do the same calculations on the global NumPy array,
    to confirm that the distributed methods give the same results.
    '''
    # Slicing.
    print('Slicing...')
    slice_volume(da, base_filename='slice_')
    # Statistics per-trace
    print('Analyzing statistics...')
    analyze_statistics(da, compare=compare, verbose=verbose)
    # 3-point average filter.
    print('Filtering avg3...')
    filtered_avg3_da = analyze_filter(da,
                                      local_filter_avg3,
                                      filter_avg3,
                                      compare=compare,
                                      verbose=verbose)
    # 3-point maximum filter.
    print('Filtering max3...')
    filtered_max3_da = analyze_filter(da,
                                      local_filter_max3,
                                      filter_max3,
                                      compare=compare,
                                      verbose=verbose)
    # Slice the filtered array.
    print('Slicing filtered array...')
    slice_volume(filtered_avg3_da, base_filename='filtered_')
    # Save filtered output to .dnpy files.
    print('Saving .dnpy files...')
    output_dnpy_filename = 'filtered_avg3'
    save_dnpy_distarray(output_dnpy_filename, filtered_avg3_da)
    # Save filtered distarray to new HDF5 file.
    print('Saving .hdf5 file...')
    output_filename = 'filtered_avg3.hdf5'
    save_hdf5_distarray(output_filename, key, filtered_avg3_da)


def main():
    # Parse arguments:
    #     --file <filename>
    #     --key <keyname>
    #     --hdf5
    #     --dnpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        default='seismic.hdf5',
                        help='Name of input file.')
    parser.add_argument('--key',
                        default='seismic',
                        help='Name of HDF5 key for data.')
    parser.add_argument('--hdf5',
                        action='store_true',
                        help='Read input from an HDF5 file.')
    parser.add_argument('--dnpy',
                        action='store_true',
                        help='Read input from .dnpy files.')
    args = parser.parse_args()
    # Extract arguments and complain about invalid ones.
    filename = args.file
    key = args.key
    use_hdf5 = args.hdf5
    use_dnpy = args.dnpy
    # Pick either HDF5 or .dnpy
    if (use_hdf5 == False) and (use_dnpy == False):
        use_hdf5 = True
    if (use_hdf5 == True) and (use_dnpy == True):
        raise ValueError('Can only specify one of --hdf5 or --dnpy.')
    # Desired distribution method.
    dist = ('b', 'b', 'n')
    # Processing options.
    compare = True
    verbose = False
    # Load the seismic volume.
    da = load_seismic_volume(filename, key, dist, use_hdf5)
    # Process the seismic volume.
    process_seismic_volume(da,
                           key=key,
                           compare=compare,
                           verbose=verbose)


if __name__ == '__main__':
    main()
    print('Done.')
