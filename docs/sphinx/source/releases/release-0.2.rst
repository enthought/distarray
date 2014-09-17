==============================================================================
DistArray 0.2: development release
==============================================================================

Documentation: http://distarray.readthedocs.org
License: Three-clause BSD
Python versions: 2.7 and 3.3
OS support: \*nix and Mac OS X

DistArray aims to bring the strengths of NumPy to data-parallel
high-performance computing. It provides distributed multi-dimensional
NumPy-like arrays and distributed ufuncs, distributed IO capabilities, and can
integrate with external distributed libraries, like Trilinos. DistArray works
with NumPy and builds on top of it in a flexible and natural way.

Brian Granger started DistArray as a NASA-funded SBIR project in 2008.
Enthought picked it up as part of a DOE Phase II SBIR [0] to provide a
generally useful distributed array package. It builds on IPython,
IPython.parallel, NumPy, MPI, and interfaces with the Trilinos suite of
distributed HPC solvers (via PyTrilinos) [1].

Distarray:

* has a client-engine (or master-worker) process design -- data resides on the
  worker processes, commands are initiated from master;
* allows full control over what is executed on the worker processes and
  integrates transparently with the master process;
* allows direct communication between workers bypassing the master process for
  scalability;
* integrates with IPython.parallel for interactive creation and exploration of
  distributed data;
* supports distributed ufuncs (currently without broadcasting);
* builds on and leverages MPI via MPI4Py in a transparent and user-friendly
  way;
* supports NumPy-like structured multidimensional arrays;
* has basic support for unstructured arrays;
* supports user-controllable array distributions across workers (block,
  cyclic, block-cyclic, and unstructured) on a per-axis basis;
* has a straightforward API to control how an array is distributed;
* has basic plotting support for visualization of array distributions;
* separates the array’s distribution from the array’s data -- useful for
  slicing, reductions, redistribution, broadcasting, all of which will be
  implemented in coming releases;
* implements distributed random arrays;
* supports .npy-like flat-file IO and hdf5 parallel IO (via h5py); leverages
  MPI-based IO parallelism in an easy-to-use and transparent way; and
* supports the distributed array protocol [2], which allows independently
  developed parallel libraries to share distributed arrays without copying,
  analogous to the PEP-3118 new buffer protocol.
* This is the first public development release. DistArray is not ready for
  real-world use, but we want to get input from the larger scientific-Python
  community to help drive its development. The API is changing rapidly and we
  are adding many new features on a fast timescale. For that reason, DistArray
  is currently implemented in pure Python for maximal flexibility. Performance
  improvements are coming.

The 0.2 release's goals are to provide the components necessary to support
upcoming features that are non-trivial to implement in a distributed
environment.

Planned features for upcoming releases:

* Distributed reductions
* Distributed slicing
* Distributed broadcasting
* Distributed fancy indexing
* Re-distribution methods
* Integration with Trilinos [1] and other packages [3] that subscribe to the
  distributed array protocol [2]
* Lazy evaluation and deferred computation for latency hiding
* Out-of-core computations
* Extensive examples, tutorials, documentation
* Support for distributed sorting and other non-trivial distributed algorithms
* MPI-only communication for non-interactive deployment on clusters and
  supercomputers
* End-user control over communication and temporary array creation, and other
  performance aspects of distributed computations

[0] http://www.sbir.gov/sbirsearch/detail/410257
[1] http://trilinos.org/
[2] http://distributed-array-protocol.readthedocs.org/en/rel-0.10.0/
[3] http://www.mcs.anl.gov/petsc/
