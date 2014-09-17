Installation
------------

DistArray requires the following Python libraries:

* `NumPy`_,
* `IPython`_, and
* `Mpi4Py`_.

.. _NumPy: http://www.numpy.org
.. _IPython: http://ipython.org
.. _Mpi4Py: http://mpi4py.scipy.org

Optionally, DistArray can make use of:

* `h5py`_ built against a parallel-enabled build of HDF5 (for HDF5 IO), and
* `matplotlib`_ (for making plots of DistArray distributions).

.. _h5py: http://www.h5py.org/
.. _matplotlib: http://matplotlib.org/

If you have the above, you should be able to install DistArray with::

    python setup.py install

or::

    pip install distarray


Testing Your Installation
-------------------------

To test your installation, you will first need to start an IPython.parallel
cluster with MPI enabled.  The easist way is to use use the ``dacluster``
command that comes with DistArray::

    dacluster start

See ``dacluster``'s help for more::
    
    dacluster --help

You should then be able to run all the tests from the DistArray source
directory with::

    make test

or from anywhere with::

    python -m distarray.run_tests
