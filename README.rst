.. Travis badge
.. image:: https://travis-ci.org/enthought/distarray.png?branch=master   
   :target: https://travis-ci.org/enthought/distarray

.. Coveralls badge
.. image:: https://coveralls.io/repos/enthought/distarray/badge.png?branch=master
   :target: https://coveralls.io/r/enthought/distarray?branch=master
   
.. All content before the next comment will be stripped off for release.
.. *** begin README content ***

DistArray
=========

*Think globally, act locally.*


DistArray provides general multidimensional NumPy-like distributed-memory
arrays for Python.  These arrays are designed to look and feel just like
`NumPy`_ arrays but to take advantage of parallel architectures with
distributed memory.  

The project is currently under heavy development and things are changing
quickly!

DistArray is targeting users who

* know and love Python and NumPy,
* want to interactively play with distributed data,
* want to run batch-oriented distributed programs,
* want an easier way to drive and coordinate existing MPI-based codes,
* have a lot of data that may already be distributed,
* want a global view ("think globally") with local control ("act locally"),
* need to tap into existing parallel libraries like Trilinos, PETSc, or
  Elemental,
* want the interactivity of IPython and the performance of MPI.

.. _NumPy: http://www.numpy.org

Please see our documentation at `readthedocs`_ (or in the `docs` directory)
for more.  Pull requests gladly accepted.

.. _readthedocs: http://distarray.readthedocs.org
