Building HDF5 and h5py for DistArray
====================================

These are notes from trying to build HDF5 1.8.12 and h5py 2.2.1 against mpi4py
1.3 and openmpi-1.6.5 on OS X 10.8.5.

HDF5
----

Download the HDF5 source (1.8.12) and configure it with parallel support.  From
the source directory::

    $ CFLAGS=-O0 CC=/Users/robertgrant/localroot/bin/mpicc ./configure --enable-shared --enable-parallel --prefix=/Users/robertgrant/localroot

The CFLAGS setting is to get around a known problem with the tests on OS X 10.8
(http://www.hdfgroup.org/HDF5/release/known_problems/).

Build it::

    $ make

Test it::

    $ make check

This produced some errors related to ph5diff, which the website claims are "not
valid errors", so I ignored them
(http://www.hdfgroup.org/HDF5/faq/parallel.html#ph5difftest).

Install HDF5::

    $ make install

h5py
----

Build h5py against this version of HDF5.  Without setting ``HDF5_DIR``, on my
system the build found Canopy's serial version of HDF5.  In the h5py source
directory::

    $ HDF5_DIR=/Users/robertgrant/localroot/ CC=mpicc python setup.py build --mpi

This gives me an error about "MPI Message" addressed here::

    https://github.com/h5py/h5py/issues/401
   
After patching api_compat.h as suggested, it builds.  One could also use the
``master`` version of h5py from GitHub instead of the latest release.

Run the tests::

    $ python setup.py test

and install h5py::

    $ python setup.py install

You should now be able to run the example code listed here::

    http://docs.h5py.org/en/latest/mpi.html#using-parallel-hdf5-from-h5py
