.. DistArray documentation master file, created by
   sphinx-quickstart on Fri Jan 31 01:11:34 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


DistArray |version|
===================

DistArray provides general multidimensional NumPy-like distributed arrays to
Python.  It intends to bring the strengths of NumPy to data-parallel
high-performance computing.  DistArray has a similar API to `NumPy`_.

The project is currently under heavy development and things are changing
quickly!

DistArray is for users who

* know and love Python and NumPy,
* want to scale NumPy to larger distributed datasets,
* want to interactively play with distributed data but also
* want to run batch-oriented distributed programs;
* want an easier way to drive and coordinate existing MPI-based codes,
* have a lot of data that may already be distributed,
* want a global view ("think globally") with local control ("act locally"),
* need to tap into existing parallel libraries like Trilinos, PETSc, or
  Elemental,
* want the interactivity of IPython and the performance of MPI.

DistArray is designed to work with other packages that implement the
`Distributed Array Protocol`_.

.. _Distributed Array Protocol: http://distributed-array-protocol.readthedocs.org
.. _NumPy: http://www.numpy.org


Installation
------------

Dependencies for DistArray:

* NumPy
* IPython
* Mpi4Py

Optional dependencies:

* For HDF5 IO: h5py built against a parallel-enabled build of HDF5
* For plotting: matplotlib

Dependencies to build the documentation:

* Sphinx
* sphinxcontrib.napoleon

If you have the above, you should be able to install this package with::

    python setup.py install

or::

    python setup.py develop


To run the tests, you will need to start an IPython.parallel cluster.  You can
use ``ipcluster``, or you can use the ``dacluster`` command which comes with
DistArray::

    dacluster start

You should then be able to run all the tests with::

    make test

To build this documentation, navigate to the ``docs`` directory and use the
Makefile there.  For example, to build the html documentation::

    make html

from the ``docs`` directory.


Getting Started
---------------

To see some initial examples of what distarray can do, check out the
``examples`` directory and our tests.  More usage examples will be forthcoming
as the API stabilizes.


History
-------

DistArray was started by Brian Granger in 2008 and is currently being developed
at Enthought by a team led by Kurt Smith, in partnership with Bill Spotz from
Sandia's (Py)Trilinos project and Brian Granger and Min RK from the IPython
project.


Other Documentation
-------------------

.. toctree::
   :maxdepth: 1

   DistArray API Reference <distarray.rst>
   The Distributed Array Protocol <http://distributed-array-protocol.readthedocs.org>
   Notes on building HDF5 and h5py <hdf5-notes.rst>
   Notes on building environment-modules <environment-modules-notes.rst>
   Licensing for bundled `six` module (Python 2 / 3 compatibility) <six-license.rst>

Release Notes
-------------
.. toctree::
   :maxdepth: 1
   :glob:

   releases/*



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

