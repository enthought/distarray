=============================
Installing `distarray`
=============================

You should be able to install this package with::

	python setup.py install
	
More details will be provided as things mature.

Details
=======

This package has the following dependencies:

	* NumPy
	* Cython
	* Mpi4Py

For both Cython and MPI4Py, you will need development versions that can be found here::

	svn checkout http://mpi4py.googlecode.com/svn/trunk/ mpi4py

and::

	hg clone http://hg.cython.org/cython-devel/ cython-devel

This version of distarray has been tested with the development versions of these
packages as of June 18th, 8 pm.  Earlier versions of Mpi4Py and Cython likely
won't work.