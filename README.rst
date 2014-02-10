.. image:: https://travis-ci.org/enthought/distarray.png
   :target: https://travis-ci.org/enthought/distarray

DistArray
=========

The DistArray package provides dense, multidimensional, distributed-memory
arrays for Python.  These arrays are designed to look and feel just like
`NumPy`_ arrays but to take advantage of parallel architectures with
distributed memory.  This project is currently under heavy development, so
things may change quickly!

DistArray is targeting users who

* want to use more than 1 node but less that 1000,
* have a lot of data that may already be distributed,
* want easy parallel computation on distributed arrays with the interactivity
  of IPython and the familiar interface of NumPy arrays.

.. _NumPy: http://www.numpy.org

Please see our documentation at `distarray.readthedocs.org` (or in the `docs`
directory) for more.  Pull requests happily accepted.
