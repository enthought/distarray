.. DistArray documentation master file, created by
   sphinx-quickstart on Fri Jan 31 01:11:34 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


DistArray |release|
===================

The DistArray package provides dense, multidimensional, distributed-memory
arrays for Python.  These arrays are designed to look and feel just like
`NumPy`_ arrays but to take advantage of parallel architectures with
distributed memory.  It is currently under heavy development, so things may
change quickly!

DistArray is targeting users who

* want to use more than 1 node but less that 1000,
* have a lot of data that may already be distributed,
* want easy parallel computation on distributed arrays with the interactivity
  of IPython and the familiar interface of NumPy arrays.

.. _NumPy: http://www.numpy.org


History
-------

DistArray was started by Brian Granger in 2008 and is currently being developed
at Enthought by a team led by Kurt Smith, in partnership with Bill Spotz from
Sandia's (Py)Trilinos project and Brian Granger and Min RK from the IPython
project.


Documentation
-------------

.. toctree::
   :maxdepth: 2

   modules




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

