conda-quickstart
================

The ``conda-quickstart`` script creates a new conda environment that includes
DistArray and its dependencies.  Currently, neither conda-quickstart (nor
DistArray) are supported on Microsoft Windows.

Depending on system hardware and the prior availability of dependencies, the
script can take anywhere from less than a minute up to a few hours to run. If
the installation is interrupted for any reason, delete the created conda
environment (``conda env remove -n <env-name>``) and re-run the script.

Prerequisites for using conda-quickstart are:

- A working Anaconda or Miniconda installation

Additionally, OSX users will need:

- A working MPI distribution that provides the ``mpicc`` compiler wrapper, *OR*
- A working copy of HomeBrew or MacPorts to install MPI (MacPorts users will
  need 'sudo' privileges)

Notes
-----

- On OSX: ``conda-quickstart`` will install Open-MPI with MacPorts or Homebrew and
  will install mpi4py using pip.
- On Linux: ``conda-quickstart`` will install MPI and mpi4py with conda
  (installs MPICH2 at the time of writing).
