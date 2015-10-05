================================================================================
		   DistArray Quickstart [OSX/Linux only]
================================================================================
The distarray/quickstart script creates a new conda environment with a working
copy of DistArray. Currently, quickstart is NOT supported on Microsoft Windows.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
Depending on system hardware and the prior availability of dependencies, the
script can take anywhere from less than a minute upto a few hours to run. If the
installation is interrupted for any reason, delete the created conda environment
[conda env remove -n <env-name>] and re-run the script.

Prerequisites for using DistArray quickstart are:
- A working conda installation (Anaconda/Miniconda)

Additionally, OSX users will need:
- A working MPI distribution that provides the 'mpicc' compiler wrapper
                                      OR
- A working copy of HomeBrew or MacPorts to install MPI (MacPorts users will
  need 'sudo' privileges)

Notes
    - OSX: Using quickstart with MacPorts/HomeBrew will build DistArray
      using the Open-MPI implementation
    - Linux: Using quickstart will build DistArray using mpi4py binaries
      from conda
--------------------------------------------------------------------------------
