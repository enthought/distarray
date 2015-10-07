Installation
------------

DistArray requires the following Python libraries:

* `numpy`_,
* `ipyparallel`_, and
* `mpi4py`_.

.. _numpy: http://www.numpy.org
.. _ipyparallel: https://github.com/ipython/ipyparallel
.. _mpi4py: http://mpi4py.scipy.org

Optionally, DistArray can make use of:

* `h5py`_ built against a parallel-enabled build of HDF5 (for HDF5 IO), and
* `matplotlib`_ (for making plots of DistArray distributions).

.. _h5py: http://www.h5py.org/
.. _matplotlib: http://matplotlib.org/

If you have the above, you should be able to install DistArray with::

    python setup.py install

or::

    pip install distarray


Experimental conda-quickstart
-----------------------------

Alternatively, we have an experimental `conda-quickstart` script in the
``quickstart`` directory of the root of this source tree.  Given an Anaconda
installation, this script will attempt to install create a conda environment
for you, installing MPI, mpi4py, and the above Python dependencies as
necessary.  See ``README.rst`` in that directory for more information.


Testing Your Installation
-------------------------

To test your installation, you will first need to start an IPython.parallel
cluster with MPI enabled.  The easist way is to use the ``dacluster`` command
that comes with DistArray::

    dacluster start

See ``dacluster``'s help for more::

    dacluster --help

You should then be able to run all the tests from the DistArray source
directory with::

    make test

or from anywhere with::

    python -m distarray.run_tests
