DistArray Overview
==================

*Think globally, act locally.*

NumPy is at the foundation of the scientific Python stack for good reason:
NumPy arrays are easy to use, they have many powerful features like ufuncs,
slicing, and broadcasting, and they work easily with external libraries.

As data sets grow and parallel hardware becomes more widely available,
wouldn't it be great if NumPy easily supported parallel execution, without
loosing its nice interface in a miasma of low-level parallel coordination?
What would that look like?

What we want is transparent distribution of NumPy arrays over the CPU,
cluster, and supercomputer.  We want to interact with distributed NumPy arrays
the way we think about them, and get the benefit of all that parallelism.  We
also want to be able to drop down a level to control what's going on at the
data-local level when performance demands it.

Such a NumPy opens doors to providing a high-level numpy-like interface to
distributed libraries like Trilinos, PETSc, Global Arrays, Elemental, and
ScaLAPACK, among others.

All this coordination has overhead, and is at risk of becoming a performance
bottlneck.  This NumPy will need a way to allow direct execution at a
data-local level.  We will also need a way to communicate directly between
local processes when needed, rather than doing everything at a global level.

This distributed NumPy should be a good citizen and work easily with regular
NumPy arrays, with MPI, with IPython parallel, and with external distributed
algorithms.

DistArray is our vision of what distributed NumPy can be.  It brings the best
parts of NumPy to data parallel computing.  We want to *think globally* about
our arrays, interacting with them as if they are just really big NumPy arrays,
all the while *acting locally* on them for performance and control.
