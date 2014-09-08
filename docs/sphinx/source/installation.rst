Installation
------------

Dependencies for DistArray:

* NumPy
* IPython
* Mpi4Py

Optional dependencies:

* For HDF5 IO: h5py built against a parallel-enabled build of HDF5
* For plotting: matplotlib

If you have the above, you should be able to install this package with::

    python setup.py install

or::

    python setup.py develop


To run the tests, you will need to start an IPython.parallel cluster.  You can
use ``ipcluster``, or you can use the ``dacluster`` command which comes with
DistArray::

    dacluster start

You should then be able to run all the tests from the DistArray source
directory with::

    make test

or from anywhere with::

    python -m distarray.run_tests
