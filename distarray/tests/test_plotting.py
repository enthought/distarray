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

# Note: This test is here instead of in distarray/plotting/tests so that it can
# be conditionally skipped.  Having it in inside plotting/ would cause a
# failure upon running the test due to the `import *` in the plotting
# directory's `__init__.py`.

import unittest

from distarray.testing import DefaultContextTestCase
from distarray.globalapi import Distribution
from distarray.testing import import_or_skip


class TestPlotting(DefaultContextTestCase):
    """Test Context methods"""

    @classmethod
    def setUpClass(cls):
        # raise a skipTest if plotting import fails
        # (because matplotlib isn't installed, probably)
        cls.plt = import_or_skip("distarray.plotting")
        super(TestPlotting, cls).setUpClass()
        cls.da = Distribution(cls.context, (64, 64))
        cls.arr = cls.context.ones(cls.da)

    def test_plot_array_distribution(self):
        # Only tests that this runs, not that it's correct
        process_coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.plt.plot_array_distribution(self.arr, process_coords)


if __name__ == '__main__':
    unittest.main(verbosity=2)
