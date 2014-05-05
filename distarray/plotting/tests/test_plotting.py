# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for plotting.py

Many of these tests require a 4-engine cluster to be running locally.  The
engines should be launched with MPI, using the MPIEngineSetLauncher.
"""

import unittest

from distarray.dist import Context, Distribution
from distarray.plotting import plotting


class TestContext(unittest.TestCase):
    """Test Context methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.da = Distribution.from_shape(cls.context, (64, 64))
        cls.arr = cls.context.ones(cls.da)

    @classmethod
    def tearDownClass(cls):
        """Close the client connections"""
        cls.context.close()

    def test_plot_array_distribution(self):
        process_coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
        plotting.plot_array_distribution(self.arr, process_coords)


if __name__ == '__main__':
    unittest.main(verbosity=2)