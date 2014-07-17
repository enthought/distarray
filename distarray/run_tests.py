# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Functions for running DistArray tests.
"""

from __future__ import print_function

import unittest
import os
import sys
import shlex
from subprocess import Popen, PIPE

import distarray


def run_ipython_parallel_tests(path):
    """Run ipython-parallel client tests."""
    test_suite = unittest.defaultTestLoader.discover(path)
    return unittest.TextTestRunner().run(test_suite)


def _process_shell_output(specific_cmd):
    """Run a command with subprocess and pass the results through to stderr."""
    proc = Popen(shlex.split(specific_cmd), stdout=PIPE, stderr=PIPE)
    while True:
        char = proc.stderr.read().decode()
        if not char:
            break
        else:
            print(char, end="")
            sys.stderr.flush()


def run_mpi_only_tests(path, nengines=4):
    """Run mpi-client tests."""
    cmd = ("mpiexec "
           "-np 1 python -m unittest discover -c -s {path} : "
           "-np {nengines} {path}/distarray/apps/engine.py")
    specific_cmd = cmd.format(nengines=nengines, path=path)
    print(specific_cmd)
    _process_shell_output(specific_cmd)


def run_local_tests(path, nengines=4):
    """Run engine-only MPI tests."""
    cmd = ("mpiexec "
           "-np {nengines} python -m unittest discover -cp "
           "paralleltest_*.py -s {path}")
    specific_cmd = cmd.format(nengines=nengines, path=path)
    print(specific_cmd)
    _process_shell_output(specific_cmd)


def test(nengines=4):
    """Run all DistArray tests."""
    path = os.path.split(os.path.split(distarray.__file__)[0])[0]
    print("IPython.parallel client tests:")
    run_ipython_parallel_tests(path)
    print()
    print("MPI client tests:")
    run_mpi_only_tests(path, nengines=nengines)
    print()
    print("Engine-only MPI tests:")
    run_local_tests(path, nengines=nengines)


if __name__ == "__main__":
    test()
