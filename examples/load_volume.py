''' Load the fake seismic volume and do stuff with it. '''

import os.path
import numpy
import h5py

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray


# Load the seismic data from the HDF5 file.

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
    print 'Getting array shape...'
    array_shape = get_hdf5_dataset_shape(pathname, key)
    # Create distribution.
    print 'Creating distribution...'
    distribution = Distribution.from_shape(context, array_shape, dist=dist)
    # Load HDF5 file into DistArray.
    print 'Loading HDF5 file...'
    distarray = context.load_hdf5(filename=pathname, distribution=distribution, key=key)
    print 'Done.'
    return distarray


def save_hdf5_distarray(filename, key, distarray):
    ''' Write a distarray to an HDF5 file. '''
    pathname = os.path.abspath(filename)
    context = distarray.context
    context.save_hdf5(pathname, distarray, key=key, mode='w')


def save_dnpy_distarray(filename, distarray):
    ''' Write a distarray to .dnpy files. '''
    pathname = os.path.abspath(filename)
    context = distarray.context
    context.save_dnpy(pathname, distarray)


# Print some stuff about the array.
# (Mostly only practical for small ones.)

def dump_distarray_info(da):
    ''' Print some stuff about the array. '''
    print 'Local Shapes:'
    localshapes = da.localshapes()
    print localshapes
    print 'Arrays Per Process:'
    ndarrays = da.get_ndarrays()
    for i, ndarray in enumerate(ndarrays):
        print 'Process:', i
        print ndarray
    print 'Full Array:'
    ndarray = da.toarray()
    print ndarray


# Calculate statistics on each trace [z slice].
# We get the same results when calculating via DistArray and via NumPy on the full array.

def calc_distributed_stats(distarray):
    ''' Calculate some statistics on each trace [z slices] in the distarray. '''
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
        print title
        for stat in stat_keys:
            print stat
            print stat_dict[stat]

    # Convert DistArray to NumPy array to allow comparison.
    for stat in stat_keys:
        distarray_stats[stat] = distarray_stats[stat].toarray()

    # The difference is ideally zero.
    if verbose: print 'Calculating difference...'
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
        print stat, 'is_close:', is_close


def analyze_statistics(distarray, compare=True, verbose=False):
    ''' Calculate statistics on each trace, via DistArray and NumPy.

    The results should match within numerical precision.
    '''
    if verbose: print 'Calculating statistics...'
    # Using DistArray methods.
    if verbose: print 'DistArray statistics...'
    distributed_stats = calc_distributed_stats(distarray)
    if compare:
        # Convert to NumPy array and use NumPy methods.
        if verbose: print 'NumPy array statistics...' 
        ndarray = distarray.tondarray()
        if verbose: print 'Converted to ndarray...'
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

    from distarray.local import LocalArray
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
        b[:, :, k] = a[:, :, k0:k1+1].max(axis=2)
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
            b[:, :, k] = a[:, :, k0:k1+1].max(axis=2)
        return b

    from distarray.local import LocalArray
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


def analyze_filter(da, local_filter, numpy_filter, compare=True, verbose=False):
    ''' Apply the filter both via DistArray methods and via NumPy methods. '''
    # Via DistArray.
    result_distarray = distributed_filter(da, local_filter)
    if verbose:
        # Print results of averaging.
        print 'Original:'
        print da.toarray()
        print 'Filtered:'
        print result_distarray.toarray()
    if compare:
        # Filter via NumPy array.
        ndarray = da.toarray()
        result_numpy = undistributed_filter(ndarray, numpy_filter)
        # Difference between DistArray and NumPy results.
        distributed_filtered = result_distarray.toarray()
        undistributed_filtered = result_numpy
        diff = distributed_filtered - undistributed_filtered
        print 'Difference of DistArray - NumPy filters:'
        if verbose:
            print diff
        is_close = numpy.allclose(distributed_filtered, undistributed_filtered)
        print 'is_close:', is_close
    # Return the filtered distarray.
    return result_distarray


# Main processing function.

def process_seismic_volume(filename, key, dist, compare=True, verbose=False):
    ''' Load the seismic volume from the specified HDF5 file,
    and do some DistArray processing with it.

    Also do the same calculations on the global NumPy array,
    to confirm that the distributed methods give the same results.
    '''
    # Create context.
    context = Context()
    # Load HDF5 file as DistArray.
    da = load_hdf5_distarray(context, filename, key, dist)
    # Print some stuff about the array.
    if False:
        dump_distarray_info(da)
    # Statistics per-trace
    analyze_statistics(da, compare=compare, verbose=verbose)
    # 3-point average filter.
    filtered_avg3_da = analyze_filter(da,
                                      local_filter_avg3,
                                      filter_avg3,
                                      compare=compare,
                                      verbose=verbose)
    # 3-point maximum filter.
    filtered_max3_da = analyze_filter(da,
                                      local_filter_max3,
                                      filter_max3,
                                      compare=compare,
                                      verbose=verbose)
    # Save filtered distarray to new HDF5 file.
    output_filename = 'filtered_avg3.hdf5'
    save_hdf5_distarray(output_filename, key, filtered_avg3_da)
    # And .dnpy files.
    output_dnpy_filename = 'filtered_avg3'
    save_dnpy_distarray(output_dnpy_filename, filtered_avg3_da)


def main():
    # Filename with data.
    #filename = 'seismic.hdf5'
    filename = 'seismic_512.hdf5'
    # Name of data block inside file.
    key = 'seismic'
    # Desired distribution method.
    dist = ('b', 'b', 'n')
    #dist = ('c', 'c', 'n')
    # Processing options.
    compare = False
    verbose = False
    process_seismic_volume(filename=filename, key=key, dist=dist, compare=compare, verbose=verbose)


if __name__ == '__main__':
    main()
    print 'Done.'
