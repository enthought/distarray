''' Load the fake seismic volume and do stuff with it. '''

import os.path

import h5py

from distarray.dist import Context, Distribution

#SIZE = (10, 10, 20)
SIZE = (4, 4, 10)

def load_volume():
    context = Context()

    # Filename for load_hdf5() needs the full path.
    filename = 'seismic.hdf5'
    pathname = os.path.abspath(filename)

    # Need to know array size.
    distribution = Distribution.from_shape(context, SIZE, dist=('b', 'b', 'n'))

    # Name of data block inside file.
    key = 'seismic'

    # Load HDF5 file into DistArray.
    da = context.load_hdf5(filename=pathname, distribution=distribution, key=key)

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
