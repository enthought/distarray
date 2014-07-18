==============================================================================
DistArray 0.4 development release
==============================================================================

**Documentation:** http://distarray.readthedocs.org

**License:** Three-clause BSD

**Python versions:** 2.7 and 3.3

**OS support:** \*nix and Mac OS X

What is DistArray?
------------------

DistArray aims to bring the strengths of NumPy to data-parallel
high-performance computing. It provides distributed multi-dimensional
NumPy-like arrays and distributed ufuncs, distributed IO capabilities, and can
integrate with external distributed libraries like Trilinos. DistArray works
with NumPy and builds on top of it in a flexible and natural way.

0.4 Release
-----------

This is the third development release. 

Noteworthy improvements in 0.4 include:

* basic slicing support;
* significant performance enhancements;
* reduction methods now support boolean arrays;
* an IPython notebook that demos basic functionality; and
* many bug fixes, API improvements, and refactorings.

DistArray is nearly ready for real-world use.  The project is evolving rapidly
and input from the larger scientific-Python community is very valuable and
helps drive development.

Existing features
-----------------

DistArray:

* has a client-engine (or master-worker) process design -- data resides on the
  worker processes, and commands are initiated from master;
* allows full control over what is executed on the worker processes and
  integrates transparently with the master process;
* allows direct communication between workers, bypassing the master process
  for scalability;
* integrates with IPython.parallel for interactive creation and exploration of
  distributed data;
* supports distributed ufuncs (currently without broadcasting);
* builds on and leverages MPI via MPI4Py in a transparent and user-friendly
  way;
* supports NumPy-like multidimensional arrays;
* has basic support for unstructured arrays;
* supports user-controllable array distributions across workers (block,
  cyclic, block-cyclic, and unstructured) on a per-axis basis;
* has a straightforward API to control how an array is distributed;
* has basic plotting support for visualization of array distributions;
* separates the array’s distribution from the array’s data -- useful for
  slicing, reductions, redistribution, broadcasting, and other operations;
* implements distributed random arrays;
* supports ``.npy``-like flat-file IO and hdf5 parallel IO (via ``h5py``);
  leverages MPI-based IO parallelism in an easy-to-use and transparent way;
  and
* supports the distributed array protocol [protocol]_, which allows
  independently developed parallel libraries to share distributed arrays
  without copying, analogous to the PEP-3118 new buffer protocol.

Planned features and roadmap
----------------------------

Near-term features and improvements include:

* MPI-only communication for performance and deployment on clusters and
  supercomputers;
* array re-distribution capabilities;
* interoperation with Trilinos [Trilinos]_;
* expanded tutorials, examples, and other introductory material; and
* distributed broadcasting support.

The longer-term roadmap includes:

* Lazy evaluation and deferred computation for latency hiding;
* Integration with other packages [petsc]_ that subscribe to the distributed
  array protocol [protocol]_;
* Distributed fancy indexing;
* Out-of-core computations;
* Support for distributed sorting and other non-trivial distributed
  algorithms; and
* End-user control over communication and temporary array creation, and other
  performance aspects of distributed computations.

History and funding
-------------------

Brian Granger started DistArray as a NASA-funded SBIR project in 2008.
Enthought picked it up as part of a DOE Phase II SBIR [SBIR]_ to provide a
generally useful distributed array package.  It builds on NumPy, MPI, MPI4Py,
IPython, IPython.parallel, and interfaces with the Trilinos suite of
distributed HPC solvers (via PyTrilinos [Trilinos]_).

This material is based upon work supported by the Department of Energy under
Award Number DE-SC0007699.

This report was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor any agency
thereof, nor any of their employees, makes any warranty, express or implied,
or assumes any legal liability or responsibility for the accuracy,
completeness, or usefulness of any information, apparatus, product, or process
disclosed, or represents that its use would not infringe privately owned
rights.  Reference herein to any specific commercial product, process, or
service by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or favoring
by the United States Government or any agency thereof.  The views and opinions
of authors expressed herein do not necessarily state or reflect those of the
United States Government or any agency thereof.


.. [protocol] http://distributed-array-protocol.readthedocs.org/en/rel-0.10.0/
.. [Trilinos] http://trilinos.org/
.. [petsc] http://www.mcs.anl.gov/petsc/
.. [SBIR] http://www.sbir.gov/sbirsearch/detail/410257

.. vim:spell
