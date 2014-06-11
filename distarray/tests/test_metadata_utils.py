# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from distarray import metadata_utils
from distarray.dist import Distribution, Context


class TestMakeGridShape(unittest.TestCase):

    def test_make_grid_shape(self):
        grid_shape = metadata_utils.make_grid_shape((20, 20), ('b', 'b'), 12)
        self.assertEqual(grid_shape, (3, 4))


class TestPositivify(unittest.TestCase):

    def test_positive_index(self):
        result = metadata_utils.positivify(5, 10)
        self.assertEqual(result, 5)

    def test_negative_index(self):
        result = metadata_utils.positivify(-2, 10)
        self.assertEqual(result, 8)


class TestGridSizes(unittest.TestCase):

    def test_dist_sizes(self):
        context = Context()
        dist = Distribution.from_shape(context, (2, 3, 4),
                                       dist=('n', 'b', 'c'))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.get_shape_from_dim_data_per_rank(ddpr)
        if len(context.view) == 4:
            self.assertEqual(shapes, [(2, 2, 2), (2, 2, 2), (2, 1, 2),
                                      (2, 1, 2)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
