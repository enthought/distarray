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

def calc_distributed_stats(distarray, verbose=False):
    ''' Calculate some statistics on each trace [z slices] in the distarray. '''
    if verbose: print 'min...' 
    trace_min = distarray.min(axis=2).toarray()
    if verbose: print 'max...' 
    trace_max = distarray.max(axis=2).toarray()
    if verbose: print 'mean...' 
    trace_mean = distarray.mean(axis=2).toarray()
    if verbose: print 'var...' 
    trace_var = distarray.var(axis=2).toarray()
    if verbose: print 'std...' 
    trace_std = distarray.std(axis=2).toarray()
    # Collect into dict.
    distributed_stats = {
        'min': trace_min,
        'max': trace_max,
        'mean': trace_mean,
        'var': trace_var,
        'std': trace_std,
    }
    return distributed_stats


def calc_undistributed_stats(ndarray, verbose=False):
    ''' Calculate some statistics on each trace [z slices] in the ndarray. '''
    # Statistics per-trace
    if verbose: print 'NumPy array statistics...' 
    if verbose: print 'min...' 
    trace_min = ndarray.min(axis=2)
    if verbose: print 'max...' 
    trace_max = ndarray.max(axis=2)
    if verbose: print 'mean...' 
    trace_mean = ndarray.mean(axis=2)
    if verbose: print 'var...' 
    trace_var = ndarray.var(axis=2)
    if verbose: print 'std...' 
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


def compare_stats(distributed_stats, undistributed_stats, verbose=False):
    ''' Compare the statistics made on DistArray vs NumPy array. '''

    stat_keys = ['min', 'max', 'mean', 'var', 'std']

    def print_stats(title, stat_dict):
        ''' Print out the statistics. '''
        print title
        for stat in stat_keys:
            print stat
            print stat_dict[stat]

    # The difference is ideally zero.
    if verbose: print 'Calculating difference...'
    difference_stats = {}
    for stat in distributed_stats:
        diff = distributed_stats[stat] - undistributed_stats[stat]
        difference_stats[stat] = diff

    # Print statistics.
    if verbose:
        print_stats('Distributed stats:', distributed_stats)
        print_stats('Undistributed stats:', undistributed_stats)
        print_stats('Difference:', difference_stats)

    # Test with allclose().
    for stat in stat_keys:
        distributed_stat = distributed_stats[stat]
        undistributed_stat = undistributed_stats[stat]
        # For 512x512x1024, single precision, var has errors which exceed
        # the default allclose() bounds, many about 0.004.
        is_close = numpy.allclose(distributed_stat, undistributed_stat)
        print stat, 'is_close:', is_close


def analyze_statistics(distarray, verbose=False):
    ''' Calculate statistics on each trace, via DistArray and NumPy.

    The results should match within numerical precision.
    '''
    if verbose: print 'Calculating statistics...'
    # Using DistArray methods.
    if verbose: print 'DistArray statistics...'
    distributed_stats = calc_distributed_stats(distarray, verbose=verbose)
    # Convert to NumPy array and use NumPy methods.
    if verbose: print 'NumPy array statistics...' 
    ndarray = distarray.tondarray()
    if verbose: print 'Converted to ndarray...'
    undistributed_stats = calc_undistributed_stats(ndarray, verbose=verbose)
    # Compare.
    compare_stats(distributed_stats, undistributed_stats, verbose=verbose)


# 3-point averaging filter. This uses a 2-point average on the ends.

def filter_3(a):
    ''' Filter a local array via 3-point average over z axis.
    A 2-point average is used at the ends. '''
    from numpy import empty_like
    b = empty_like(a)
    b[:, :, 0] = (a[:, :, 0] + a[:, :, 1]) / 2.0
    b[:, :, 1:-1] = (a[:, :, :-2] + a[:, :, 1:-1] + a[:, :, 2:]) / 3.0
    b[:, :, -1] = (a[:, :, -2] + a[:, :, -1]) / 2.0
    return b


def local_filter_3(la):
    ''' Filter a local array via 3-point average over z axis. '''

    def filter_3(a):
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
    b = filter_3(a)
    res = LocalArray(la.distribution, buf=b)
    rtn = proxyize(res)
    return rtn


def analyze_filter(context, da):
    ''' Apply the filter both via DistArray methods and via NumPy methods. '''
    # Via DistArray.
    res_key = context.apply(local_filter_3, (da.key,))
    res_da = DistArray.from_localarrays(res_key[0], context=context)
    res_nd = res_da.toarray()
    # Filter via NumPy array.
    nd = da.toarray()
    res2_nd = filter_3(da.toarray())
    # Print results of averaging.
    print 'Original:'
    print da.toarray()
    print 'Averaged:'
    print res_nd
    # Difference between DistArray and NumPy results.
    distributed_filtered = res_nd
    undistributed_filtered = res2_nd
    diff = distributed_filtered - undistributed_filtered
    print 'Difference of DistArray - NumPy filters:'
    print diff
    is_close = numpy.allclose(distributed_filtered, undistributed_filtered)
    print 'is_close:', is_close


#

def load_volume():
    # Filename with data.
    filename = 'seismic.hdf5'
    #filename = 'seismic_512.hdf5'
    # Name of data block inside file.
    key = 'seismic'
    # Desired distribution method.
    #dist = ('b', 'b', 'n')
    dist = ('c', 'c', 'n')
    # Create context.
    context = Context()
    # Load HDF5 file as DistArray.
    da = load_hdf5_distarray(context, filename, key, dist)
    # Print some stuff about the array.
    if False:
        dump_distarray_info(da)
    # Statistics per-trace
    analyze_statistics(da, verbose=True)
    # 3-point filter.
    analyze_filter(context, da)


def main():
    load_volume()


if __name__ == '__main__':
    main()
    print 'Done.'
