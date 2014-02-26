.. DistArray documentation master file, created by
   sphinx-quickstart on Fri Jan 31 01:11:34 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


DistArray |release|
===================

The DistArray package provides dense, multidimensional, distributed-memory
arrays for Python.  These arrays are designed to look and feel like `NumPy`_
arrays but to take advantage of parallel architectures with distributed memory.
It is currently under heavy development, so things may change quickly!

DistArray is targeting users who

* want to use more than 1 node but less that 1000,
* have a lot of data that may already be distributed,
* want easy parallel computation on distributed arrays with the interactivity
  of IPython and the familiar interface of NumPy arrays.

.. _NumPy: http://www.numpy.org


Installation
------------

Dependencies for DistArray:

* NumPy
* IPython
* Mpi4Py
* six

Dependencies for optional HDF5 IO:

* h5py built against a parallel-enabled build of HDF5

Dependencies to build the documentation:

* Sphinx
* sphinxcontrib.napoleon

If you have the above, you should be able to install this package with::

    python setup.py install

or::

    python setup.py develop


To run the tests, you will need to start an IPython.parallel cluster with at
least four engines, for example::

    ipcluster start -n4 --engines=MPI

or under Python 3::

    ipcluster3 start -n4 --engines=MPI

You should then be able to run all the tests with::

    make test

To build this documentation, navigate to the `docs` directory and use the
Makefile there.  For example, to build the html documentation::

    make html

from the `docs` directory.


History
-------

DistArray was started by Brian Granger in 2008 and is currently being developed
at Enthought by a team led by Kurt Smith, in partnership with Bill Spotz from
Sandia's (Py)Trilinos project and Brian Granger and Min RK from the IPython
project.


Other Documentation
-------------------

.. toctree::
   :maxdepth: 2

   Notes on building environment-modules <environment-modules-notes.rst>
   Notes on building HDF5 and h5py <hdf5-notes.rst>
   API Reference <modules.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

