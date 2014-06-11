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

    @classmethod
    def setUpClass(cls):
        cls.context = Context()

    def test_dist_sizes(self):
        dist = Distribution.from_shape(self.context, (2, 3, 4),
                                       dist=('n', 'b', 'c'))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.shapes_from_dim_data_per_rank(ddpr)
        if len(self.context.view) == 4:
            self.assertEqual(shapes, [(2, 2, 2), (2, 2, 2), (2, 1, 2),
                                      (2, 1, 2)])

    def test_n_size(self):
        dim_dict = {'dist_type': 'n',
                    'size': 42,
                    'proc_grid_size': 1,
                    'proc_grid_rank': 0}

        dist = Distribution(self.context, (dim_dict,))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.shapes_from_dim_data_per_rank(ddpr)
        self.assertEqual(shapes, [(42,)])

    def test_b_size(self):
        dim_dict = {'dist_type': 'b',
                    'size': 42,
                    'bounds': [0, 20, 42],
                    'proc_grid_size': 2,
                    'proc_grid_rank': 0,
                    'start': 0,
                    'stop': 42}
        dist = Distribution(self.context, (dim_dict,))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.shapes_from_dim_data_per_rank(ddpr)
        self.assertEqual(shapes, [(20,), (22,)])

    def test_c_size(self):
        dim_dict = {'dist_type': 'c',
                    'size': 42,
                    'proc_grid_size': 2,
                    'proc_grid_rank': 0,
                    'start': 0}
        dist = Distribution(self.context, (dim_dict,))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.shapes_from_dim_data_per_rank(ddpr)
        self.assertEqual(shapes, [(21,), (21,)])

    def test_bc_size(self):
        dim_dict = {'dist_type': 'b',
                    'size': 42,
                    'block_size': 2,
                    'bounds': [0, 20, 42],
                    'proc_grid_size': 2,
                    'proc_grid_rank': 0,
                    'start': 0}
        dist = Distribution(self.context, (dim_dict,))
        ddpr = dist.get_dim_data_per_rank()
        shapes = metadata_utils.shapes_from_dim_data_per_rank(ddpr)
        self.assertEqual(shapes, [(20,), (22,)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
