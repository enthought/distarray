# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
import numpy as np

from distarray.externals.six.moves import reduce

from distarray.testing import ParallelTestCase
from distarray.localapi.maps import Distribution


dd0 = dict(dist_type='n',
           size=20)
dd1 = dict(dist_type='b',
           size=39,
           start=0,
           stop=39,
           proc_grid_rank=0,
           proc_grid_size=1)
dd2 = dict(dist_type='c',
           start=1,
           size=16,
           proc_grid_rank=1,
           proc_grid_size=2)
dd3 = dict(dist_type='c',
           start=2,
           size=16,
           block_size=2,
           proc_grid_rank=1,
           proc_grid_size=2)
test_dim_data = (dd0, dd1, dd2, dd3)


class TestDistributionCreation(ParallelTestCase):

    def test_creation(self):
        distribution = Distribution(comm=self.comm,
                                dim_data=test_dim_data)
        self.assertTrue(len(distribution) == len(test_dim_data))


class TestFromShape(ParallelTestCase):

    """Is the __init__ method working properly?"""

    def test_basic_1d(self):
        """Test basic Distribution.from_shape creation."""
        self.dist_1d = Distribution.from_shape(comm=self.comm,
                                    shape=(7,), grid_shape=(4,))
        self.assertEqual(self.dist_1d.global_shape, (7,))
        self.assertEqual(self.dist_1d.dist, ('b',))
        self.assertEqual(self.dist_1d.grid_shape, (4,))
        self.assertEqual(self.dist_1d.base_comm, self.comm)
        self.assertEqual(self.dist_1d.comm_size, 4)
        self.assertTrue(self.dist_1d.comm_rank in range(4))
        self.assertEqual(self.dist_1d.comm.Get_topo(),
                         (list(self.dist_1d.grid_shape),
                          [0], list(self.dist_1d.cart_coords)))
        self.assertEqual(len(self.dist_1d), 1)
        self.assertEqual(self.dist_1d.global_shape, (7,))
        if self.dist_1d.comm_rank == 3:
            self.assertEqual(self.dist_1d.local_shape, (1,))
        else:
            self.assertEqual(self.dist_1d.local_shape, (2,))

    def test_basic_2d(self):
        """Test basic LocalArray creation."""
        self.dist_2d = Distribution.from_shape(comm=self.comm, shape=(16, 16),
                                               grid_shape=(4, 1))
        self.assertEqual(self.dist_2d.global_shape, (16, 16))
        self.assertEqual(self.dist_2d.dist, ('b', 'b'))
        self.assertEqual(self.dist_2d.grid_shape, (4, 1))
        self.assertEqual(self.dist_2d.base_comm, self.comm)
        self.assertEqual(self.dist_2d.comm_size, 4)
        self.assertTrue(self.dist_2d.comm_rank in range(4))
        self.assertEqual(self.dist_2d.comm.Get_topo(),
                         (list(self.dist_2d.grid_shape),
                          [0, 0], list(self.dist_2d.cart_coords)))
        self.assertEqual(len(self.dist_2d), 2)
        self.assertEqual(self.dist_2d.grid_shape, (4, 1))
        self.assertEqual(self.dist_2d.global_shape, (16, 16))
        self.assertEqual(self.dist_2d.local_shape, (4, 16))
        self.assertEqual(self.dist_2d.local_size,
                         np.array(self.dist_2d.local_shape).prod())

    def test_bad_distribution(self):
        """Test that invalid distribution type fails as expected."""
        with self.assertRaises(ValueError):
            # Invalid distribution type 'x'.
            Distribution.from_shape(comm=self.comm, shape=(7,),
                                    dist={0: 'x'}, grid_shape=(4,))

    def test_no_grid_shape(self):
        """Create array init when passing no grid_shape."""
        dist_nogrid = Distribution.from_shape(comm=self.comm, shape=(7,))
        grid_shape = dist_nogrid.grid_shape
        # For 1D array as created here, we expect grid_shape
        # to just be the number of engines as a tuple.
        max_size = self.comm.Get_size()
        expected_grid_shape = (max_size,)
        self.assertEqual(grid_shape, expected_grid_shape)

    def test_cart_coords(self):
        """Test getting the cart_coords attribute."""
        self.dist_1d = Distribution.from_shape(comm=self.comm, shape=(7,),
                                               grid_shape=(4,))
        actual_1d = self.dist_1d.cart_coords
        expected_1d = tuple(self.dist_1d.comm.Get_coords(self.dist_1d.comm_rank))
        self.assertEqual(actual_1d, expected_1d)

        self.dist_2d = Distribution.from_shape(comm=self.comm, shape=(16, 16),
                                               grid_shape=(4, 1))
        actual_2d = self.dist_2d.cart_coords
        expected_2d = tuple(self.dist_2d.comm.Get_coords(self.dist_2d.comm_rank))
        self.assertEqual(actual_2d, expected_2d)


class TestInitShapeEquivalence(ParallelTestCase):

    comm_size = 2

    def assert_alike(self, d0, d1):
        self.assertEqual(d0.global_shape, d1.global_shape)
        self.assertEqual(d0.dist, d1.dist)
        self.assertEqual(d0.grid_shape, d1.grid_shape)
        self.assertEqual(d0.base_comm, d1.base_comm)
        self.assertEqual(d0.comm_size, d1.comm_size)
        self.assertEqual(d0.comm_rank, d1.comm_rank)
        self.assertEqual(len(d0), len(d1))
        self.assertEqual(d0.grid_shape, d1.grid_shape)
        self.assertEqual(d0.local_shape, d1.local_shape)
        self.assertEqual(d0.local_size, d1.local_size)

    def test_block(self):
        dim00 = {
            "dist_type": 'b',
            "size": 16,
            "start": 0,
            "stop": 8,
            "proc_grid_rank": 0,
            "proc_grid_size": 2,
            }
        dim01 = {
            "dist_type": 'n',
            "size": 16,
            }
        dd0 = (dim00, dim01)

        dim10 = {
            "dist_type": 'b',
            "size": 16,
            "start": 8,
            "stop": 16,
            "proc_grid_rank": 1,
            "proc_grid_size": 2,
            }
        dim11 = {
            "dist_type": 'n',
            "size": 16,
            }
        dd1 = (dim10, dim11)
        dim_data_per_rank = (dd0, dd1)

        d0 = Distribution(comm=self.comm,
                    dim_data=dim_data_per_rank[self.comm.Get_rank()])
        d1 = Distribution.from_shape(comm=self.comm, shape=(16, 16),
                                     dist={0: 'b'}, grid_shape=(2, 1))
        self.assert_alike(d0, d1)

    def test_cyclic(self):

        dim00 = {
            "dist_type": 'n',
            "size": 16,
            }
        dim01 = {
            "dist_type": 'c',
            "size": 16,
            "start": 0,
            "proc_grid_rank": 0,
            "proc_grid_size": 2,
            }
        dd0 = (dim00, dim01)

        dim10 = {
            "dist_type": 'n',
            "size": 16,
            }
        dim11 = {
            "dist_type": 'c',
            "size": 16,
            "start": 1,
            "proc_grid_rank": 1,
            "proc_grid_size": 2,
            }
        dd1 = (dim10, dim11)

        dim_data_per_rank = (dd0, dd1)

        larr = Distribution(comm=self.comm,
                dim_data=dim_data_per_rank[self.comm.Get_rank()])
        expected = Distribution.from_shape(comm=self.comm, shape=(16, 16),
                                           dist={1: 'c'}, grid_shape=(1, 2))

        self.assert_alike(larr, expected)


class TestGridShape(ParallelTestCase):

    comm_size = 12

    def test_grid_shape(self):
        """Test various ways of setting the grid_shape."""
        dist = Distribution.from_shape(comm=self.comm, shape=(20, 20), dist='b')
        self.assertEqual(dist.grid_shape, (12, 1))
        dist = Distribution.from_shape(comm=self.comm, shape=(2*10, 6*10),
                                       dist=('b', 'b'))
        self.assertEqual(dist.grid_shape, (2, 6))
        dist = Distribution.from_shape(comm=self.comm, shape=(6*10, 2*10),
                                       dist='bb')
        self.assertEqual(dist.grid_shape, (6, 2))
        dist = Distribution.from_shape(comm=self.comm, shape=(100, 10, 300),
                                       dist='bnc')
        self.assertEqual(dist.grid_shape, (2, 1, 6))
        dist = Distribution.from_shape(comm=self.comm, shape=(100, 50, 300),
                                       dist='bbb')
        self.assertEqual(dist.grid_shape, (2, 1, 6))
        dist = Distribution.from_shape(comm=self.comm, shape=(100, 100, 150),
                                       dist='bbb')
        self.assertEqual(dist.grid_shape, (2, 2, 3))

    def test_ones_in_grid_shape(self):
        """Test not-distributed dimensions in grid_shape."""
        dist = ('n', 'b', 'n', 'c', 'n')
        glb_shape = (2, 6, 2, 8, 2)
        grid_shape = (1, 3, 1, 4, 1)
        dist_5d = Distribution.from_shape(comm=self.comm, shape=glb_shape,
                                          grid_shape=grid_shape, dist=dist)
        self.assertEqual(dist_5d.global_shape, glb_shape)
        self.assertEqual(dist_5d.grid_shape, grid_shape)
        self.assertEqual(dist_5d.base_comm, self.comm)
        self.assertEqual(dist_5d.comm_size, 12)
        self.assertTrue(dist_5d.comm_rank in range(12))
        self.assertEqual(dist_5d.comm.Get_topo(),
                         (list(dist_5d.grid_shape),
                          [0]*5, list(dist_5d.cart_coords)))
        self.assertEqual(len(dist_5d), 5)
        self.assertEqual(dist_5d.global_shape, glb_shape)
        self.assertEqual(dist_5d.local_shape, (2, 2, 2, 2, 2))
        self.assertEqual(dist_5d.local_size,
                         reduce(int.__mul__, glb_shape) // self.comm_size)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
