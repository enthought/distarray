"""
Simple test of a parallel build of h5py (from h5py's documentation).

http://h5py.readthedocs.org/en/latest/mpi.html#using-parallel-hdf5-from-h5py

If you've built h5py properly against a parallel build of hdf5, you should be
able to run this code with::

    $ mpiexec -n 4 python test_h5py.py

and then check the output with `h5dump`::

    $ h5dump parallel_test.hdf5
    HDF5 "parallel_test.hdf5" {
    GROUP "/" {
    DATASET "test" {
        DATATYPE  H5T_STD_I32LE
        DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
        DATA {
        (0): 0, 1, 2, 3
        }
    }
    }
    }
"""

from mpi4py import MPI
import h5py

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)

dset = f.create_dataset('test', (4,), dtype='i')
dset[rank] = rank

f.close()
