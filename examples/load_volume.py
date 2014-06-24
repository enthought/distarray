''' Load the fake seismic volume and do stuff with it. '''

import os.path
import numpy
import h5py

from distarray.dist import Context, Distribution


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
    # (Mostly only practical for small ones.)
    if False:
        print 'Local Shapes:'
        localshapes = da.localshapes()
        print localshapes

    if False:
        print 'Arrays Per Process:'
        ndarrays = da.get_ndarrays()
        for i, ndarray in enumerate(ndarrays):
            print 'Process:', i
            print ndarray

    if False:
        print 'Full Array:'
        ndarray = da.toarray()
        print ndarray

    # Statistics per-trace
    analyze_statistics(da, verbose=True)


def main():
    load_volume()


if __name__ == '__main__':
    main()
    print 'Done.'
