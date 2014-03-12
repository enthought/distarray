# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import unittest

from distarray.externals.six.moves import range
from distarray.local.base import BaseLocalArray


class TestBaseLocalArray(unittest.TestCase):

    def test_block(self):
        dim0 = {
            "dist_type": 'b',
            "size": 16,
            "proc_grid_size": 1,
            "proc_grid_rank": 0,
            "start": 0,
            "stop": 16,
            }

        dim1 = {
            "dist_type": 'n',
            "size": 16,
            }

        dim_data = (dim0, dim1)

        larr = BaseLocalArray(dim_data)

        self.assertEqual(larr.grid_shape, (1,))
        self.assertEqual(larr.global_shape, (16, 16))
        self.assertEqual(len(larr.maps), 2)
        self.assertEqual(larr.local_array.shape, larr.global_shape)
        self.assertEqual(larr.ndim, 2)
        self.assertEqual(larr.size, 16*16)
        self.assertEqual(larr.dist, ('b', 'n'))
        self.assertEqual(larr.distdims, (0,))
        self.assertEqual(larr.ndistdim, 1)
        self.assertEqual(larr.local_size, 16*16)

        self.assertEqual([x for x in larr.maps[0].global_index],
                         [x for x in range(16)])

    def test_cyclic(self):
        dim0 = {
            "dist_type": 'c',
            "size": 16,
            "proc_grid_size": 1,
            "proc_grid_rank": 0,
            "start": 0,
            }

        dim1 = {
            "dist_type": 'b',
            "size": 16,
            "proc_grid_size": 1,
            "proc_grid_rank": 0,
            "start": 0,
            "stop": 16,
            }

        dim_data = (dim0, dim1)

        larr = BaseLocalArray(dim_data)

        self.assertEqual(larr.grid_shape, (1, 1))
        self.assertEqual(larr.global_shape, (16, 16))
        self.assertEqual(len(larr.maps), 2)
        self.assertEqual(larr.local_array.shape, larr.global_shape)
        self.assertEqual(larr.ndim, 2)
        self.assertEqual(larr.size, 16*16)
        self.assertEqual(larr.dist, ('c', 'b'))
        self.assertEqual(larr.distdims, (0, 1))
        self.assertEqual(larr.ndistdim, 2)
        self.assertEqual(larr.local_size, 16*16)

        self.assertEqual([x for x in larr.maps[0].global_index],
                         [x for x in range(0, 16, 1)])
