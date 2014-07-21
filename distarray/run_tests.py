# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Functions for running DistArray tests.
"""

from __future__ import print_function

import os
import sys
import shlex
import subprocess

import distarray


def _run_shell_command(specific_cmd):
    """Run a command with subprocess and pass the results through to stdout.

    First, change directory to the project directory.
    """
    path = os.path.split(os.path.split(distarray.__file__)[0])[0]
    os.chdir(path)
    proc = subprocess.Popen(shlex.split(specific_cmd),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    while True:
        char = proc.stdout.read(1).decode()
        if not char:
            return proc.wait()
        else:
            print(char, end="")
            sys.stdout.flush()


def test():
    """Run all DistArray tests."""
    cmd = "make test"
    return _run_shell_command(cmd)


if __name__ == "__main__":
    sys.exit(test())
