# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for distarray's global context.

Many of these tests require a 4-engine cluster to be running locally.  The
engines should be launched with MPI, using the MPIEngineSetLauncher.

"""

import unittest

from distarray.context import Context
from distarray.client import DistArray
from distarray.creation import empty, ones, zeros
from distarray.world import WORLD


class TestWorld(unittest.TestCase):
    """Test global world context."""
    def test_world_exists(self):
        self.assertIsInstance(WORLD, Context)

    def test_world_works(self):
        a = empty((2, 2))
        b = ones((2, 2))
        c = zeros((2, 2))

        self.assertIsInstance(a, DistArray)
        self.assertIsInstance(b, DistArray)
        self.assertIsInstance(c, DistArray)


if __name__ == '__main__':
    unittest.main(verbosity=2)
