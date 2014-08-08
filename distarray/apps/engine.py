#!/usr/bin/env python
# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Script for facilitating MPI-only mode.

Starts an MPI-process-based engine.
"""

from distarray.mpi_engine import Engine

if __name__ == '__main__':
    # use argparse to generate usage text (-h)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    Engine()
