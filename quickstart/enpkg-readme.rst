enpkg-quickstart
================

[OS X only]

Note: this script is currently experimental.

The ``enpkg-quickstart`` script attempts to install DistArray, its
dependencies, and the dependencies needed to build the DistArray docs.  This
script is intended to work with your Enthought Canopy installation.

This script does *not* attempt to install parallel versions of hdf5 or h5py,
which are optional dependencies.

``enpkg-quickstart`` is intended to be run from within this directory.

Depending on your system hardware and the prior availability of dependencies,
the script can take anywhere from less than a minute up to a few hours to run.

Prerequisites
-------------

Prerequisites for using ``enpkg-quickstart`` are:

- A working Canopy or EPD installation, and

- A working MPI distribution that provides the ``mpicc`` compiler wrapper, *OR*
  a working copy of HomeBrew or MacPorts to install MPI (MacPorts users will
  need 'sudo' privileges)


Notes
-----

``enpkg-quickstart`` will install

- Open MPI with MacPorts or Homebrew, if ``mpicc`` isn't found,
- several Python dependencies using ``enpkg``, and finally
- a couple of Python dependencies (those not installable with enpkg) through
  ``pip``.
