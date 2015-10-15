conda-quickstart
================

[OS X or Linux]

Note: this script is currently *experimental*.

The ``conda-quickstart`` script attempts to create a new conda environment that
includes DistArray, its dependencies, and the dependencies required to build
the DistArray docs.

This script does *not* attempt to install parallel versions of hdf5 or h5py,
which are optional dependencies.

``conda-quickstart`` is intended to be run from within this directory

Depending on system hardware and the prior availability of dependencies, the
script can take anywhere from less than a minute up to a few hours to run. If
the installation is interrupted for any reason, delete the created conda
environment (``conda env remove -n <env-name>``) and re-run the script.

Prerequisites
-------------

Prerequisites for using ``conda-quickstart`` are:

- A working Anaconda or Miniconda installation

Additionally, OSX users will need:

- A working MPI distribution that provides the ``mpicc`` compiler wrapper
- *OR* a working copy of HomeBrew or MacPorts to install MPI (MacPorts users will
  need 'sudo' privileges)

Notes on OS X
-------------

On OSX, ``conda-quickstart`` will install

- Open MPI with MacPorts or Homebrew, if ``mpicc`` isn't found,
- several Python dependencies using ``conda``, and finally
- a couple of Python dependencies (those not installable with ``conda``)
  through ``pip``.

Notes on Linux
--------------

On Linux, ``conda-quickstart`` will install

- Several Python dependencies using ``conda`` (including MPICH2 and mpi4py),
  and
- a couple of Python dependencies (those not installable with ``conda``)
  through ``pip``.
