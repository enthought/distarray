''' Load the fake seismic volume and do stuff with it. '''

import os.path

import h5py

from distarray.dist import Context, Distribution


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
    array_shape = get_hdf5_dataset_shape(pathname, key)
    # Create distribution.
    distribution = Distribution.from_shape(context, array_shape, dist=dist)
    # Load HDF5 file into DistArray.
    distarray = context.load_hdf5(filename=pathname, distribution=distribution, key=key)
    return distarray


def load_volume():
    # Filename with data.
    filename = 'seismic.hdf5'
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
    if True:
        print 'Local Shapes:'
        localshapes = da.localshapes()
        print localshapes

    if True:
        print 'Arrays Per Process:'
        ndarrays = da.get_ndarrays()
        for i, ndarray in enumerate(ndarrays):
            print 'Process:', i
            print ndarray

    if True:
        print 'Full Array:'
        ndarray = da.toarray()
        print ndarray

    # Statistics per-trace
    trace_min = da.min(axis=2)
    trace_max = da.max(axis=2)
    trace_mean = da.mean(axis=2)
    trace_var = da.var(axis=2)
    trace_std = da.std(axis=2)

    def print_stat(name, distarray):
        print 'trace', name
        print distarray
        ndarray = distarray.toarray()
        print ndarray
    
    print_stat('min', trace_min)
    print_stat('max', trace_max)
    print_stat('mean', trace_mean)
    print_stat('var', trace_var)
    print_stat('std', trace_std)


def main():
    load_volume()


if __name__ == '__main__':
    main()
    print 'Done.'
