import unittest

import numpy as np
import matplotlib as mpl

from distarray import Context
# to be tested:
from distarray.plotting.plotting import (_get_ranks, cmap_discretize,
                                         plot_array_distribution)


class TestPlotting(unittest.TestCase):

    def test__get_ranks(self):
        context = Context(targets=range(4))
        a = context.empty((4,))
        b = _get_ranks(a).toarray()
        np.testing.assert_array_equal(b, np.arange(4))

    def test_cmap_discretize(self):
        """Smoke test for `cmap_discretize`."""
        cmap_discretize(mpl.cm.jet, 6)

    def test_plot_array_distribution(self):
        """Smoke test for `plot_array_distribution`."""
        context = Context()
        a = context.empty((4, 3))
        plot_array_distribution(a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
