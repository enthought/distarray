# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from distarray import utils
from distarray.testing import (ParallelTestCase, assert_localarrays_allclose,
                               assert_localarrays_equal)
from distarray.localapi.localarray import LocalArray, ndenumerate, ones
from distarray.localapi.maps import Distribution
from distarray.localapi.error import InvalidDimensionError, IncompatibleArrayError


class TestInit(ParallelTestCase):

    """Is the __init__ method working properly?"""

    def setUp(self):
        self.dist_1d = Distribution.from_shape(comm=self.comm,
                                        shape=(7,), grid_shape=(4,))
        self.larr_1d = LocalArray(self.dist_1d, buf=None)

        self.dist_2d = Distribution.from_shape(comm=self.comm,
                                        shape=(16, 16), grid_shape=(4, 1))
        self.larr_2d = LocalArray(self.dist_2d, buf=None)

    def test_basic_1d(self):
        """Test basic LocalArray creation."""
        self.assertEqual(self.larr_1d.global_shape, (7,))
        self.assertEqual(self.larr_1d.dist, ('b',))
        self.assertEqual(self.larr_1d.grid_shape, (4,))
        self.assertEqual(self.larr_1d.comm_size, 4)
        self.assertTrue(self.larr_1d.comm_rank in range(4))
        self.assertEqual(len(self.larr_1d.distribution), 1)
        self.assertEqual(self.larr_1d.global_shape, (7,))
        if self.larr_1d.comm_rank == 3:
            self.assertEqual(self.larr_1d.local_shape, (1,))
        else:
            self.assertEqual(self.larr_1d.local_shape, (2,))
        self.assertEqual(self.larr_1d.ndarray.shape, self.larr_1d.local_shape)
        self.assertEqual(self.larr_1d.ndarray.size, self.larr_1d.local_size)
        self.assertEqual(self.larr_1d.local_size, self.larr_1d.local_shape[0])
        self.assertEqual(self.larr_1d.ndarray.dtype, self.larr_1d.dtype)

    def test_basic_2d(self):
        """Test basic LocalArray creation."""
        self.assertEqual(self.larr_2d.global_shape, (16, 16))
        self.assertEqual(self.larr_2d.dist, ('b', 'b'))
        self.assertEqual(self.larr_2d.grid_shape, (4, 1))
        self.assertEqual(self.larr_2d.comm_size, 4)
        self.assertTrue(self.larr_2d.comm_rank in range(4))
        self.assertEqual(len(self.larr_2d.distribution), 2)
        self.assertEqual(self.larr_2d.grid_shape, (4, 1))
        self.assertEqual(self.larr_2d.global_shape, (16, 16))
        self.assertEqual(self.larr_2d.local_shape, (4, 16))
        self.assertEqual(self.larr_2d.local_size,
                         np.array(self.larr_2d.local_shape).prod())
        self.assertEqual(self.larr_2d.ndarray.shape, self.larr_2d.local_shape)
        self.assertEqual(self.larr_2d.ndarray.size, self.larr_2d.local_size)
        self.assertEqual(self.larr_2d.ndarray.dtype, self.larr_2d.dtype)

    def test_localarray(self):
        """Can the ndarray be set and get?"""
        self.larr_2d.ndarray
        la = np.random.random(self.larr_2d.local_shape)
        la = np.asarray(la, dtype=self.larr_2d.dtype)
        self.larr_2d.ndarray = la
        self.larr_2d.ndarray

    def test_bad_localarray(self):
        """ Test that setting a bad local array fails as expected. """
        self.larr_1d.ndarray
        local_shape = self.larr_1d.local_shape
        # Double dimension sizes to make an invalid shape.
        bad_shape = tuple(2 * size for size in local_shape)
        la = np.random.random(bad_shape)
        la = np.asarray(la, dtype=self.larr_1d.dtype)
        with self.assertRaises(ValueError):
            self.larr_1d.ndarray = la

    def test_cart_coords(self):
        """Test getting the cart_coords attribute"""
        actual_1d = self.larr_1d.cart_coords
        expected_1d = tuple(self.larr_1d.distribution.cart_coords)
        self.assertEqual(actual_1d, expected_1d)
        actual_2d = self.larr_2d.cart_coords
        expected_2d = tuple(self.larr_2d.distribution.cart_coords)
        self.assertEqual(actual_2d, expected_2d)


class TestLocalInd(ParallelTestCase):

    """Test the computation of local indices."""

    def test_block_simple(self):
        """Can we compute local indices for a block distribution?"""
        distribution = Distribution.from_shape(comm=self.comm, shape=(4, 4))
        la = LocalArray(distribution)
        self.assertEqual(la.global_shape, (4, 4))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(la.local_shape, (1, 4))
        row_result = [(0, 0), (0, 1), (0, 2), (0, 3)]

        row = la.comm_rank
        calc_row_result = [la.local_from_global((row, col)) for col in
                           range(la.global_shape[1])]
        self.assertEqual(row_result, calc_row_result)

    def test_block_complex(self):
        """Can we compute local indices for a block distribution?"""
        distribution = Distribution.from_shape(comm=self.comm, shape=(8, 2))
        la = LocalArray(distribution)
        self.assertEqual(la.global_shape, (8, 2))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(la.local_shape, (2, 2))
        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (1, 0), (1, 1)]
        elif la.comm_rank == 1:
            gis = [(2, 0), (2, 1), (3, 0), (3, 1)]
        elif la.comm_rank == 2:
            gis = [(4, 0), (4, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 3:
            gis = [(6, 0), (6, 1), (7, 0), (7, 1)]

        result = [la.local_from_global(gi) for gi in gis]
        self.assertEqual(result, expected_lis)

    def test_cyclic_simple(self):
        """Can we compute local indices for a cyclic distribution?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(10,), dist={0: 'c'})
        la = LocalArray(distribution)
        self.assertEqual(la.global_shape, (10,))
        self.assertEqual(la.grid_shape, (4,))

        if la.comm_rank == 0:
            gis = (0, 4, 8)
            self.assertEqual(la.local_shape, (3,))
            calc_result = [la.local_from_global((gi,)) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 1:
            gis = (1, 5, 9)
            self.assertEqual(la.local_shape, (3,))
            calc_result = [la.local_from_global((gi,)) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 2:
            gis = (2, 6)
            self.assertEqual(la.local_shape, (2,))
            calc_result = [la.local_from_global((gi,)) for gi in gis]
            result = [(0,), (1,)]
        elif la.comm_rank == 3:
            gis = (3, 7)
            self.assertEqual(la.local_shape, (2,))
            calc_result = [la.local_from_global((gi,)) for gi in gis]
            result = [(0,), (1,)]

        self.assertEqual(result, calc_result)

    def test_cyclic_complex(self):
        """Can we compute local indices for a cyclic distribution?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(8, 2), dist={0: 'c'})
        la = LocalArray(distribution)
        self.assertEqual(la.global_shape, (8, 2))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(la.local_shape, (2, 2))

        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (4, 0), (4, 1)]
        elif la.comm_rank == 1:
            gis = [(1, 0), (1, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 2:
            gis = [(2, 0), (2, 1), (6, 0), (6, 1)]
        elif la.comm_rank == 3:
            gis = [(3, 0), (3, 1), (7, 0), (7, 1)]

        result = [la.local_from_global(gi) for gi in gis]
        self.assertEqual(result, expected_lis)


class TestGlobalInd(ParallelTestCase):

    """Test the computation of global indices."""

    def round_trip(self, la):
        for indices in utils.multi_for([range(s) for s in la.local_shape]):
            gi = la.global_from_local(indices)
            li = la.local_from_global(gi)
            self.assertEqual(li, indices)

    def test_block(self):
        """Can we go from global to local indices and back for block?"""
        distribution = Distribution.from_shape(comm=self.comm, shape=(4, 4))
        la = LocalArray(distribution)
        self.round_trip(la)

    def test_cyclic(self):
        """Can we go from global to local indices and back for cyclic?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(8, 8), dist=('c', 'n'))
        la = LocalArray(distribution)
        self.round_trip(la)

    def test_crazy(self):
        """Can we go from global to local indices and back for a complex case?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(10, 100, 20),
                                        dist=('b', 'c', 'n'))
        la = LocalArray(distribution)
        self.round_trip(la)

    def test_global_limits_block(self):
        """Find the boundaries of a block distribution"""
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), dist=('b', 'n'))
        a = LocalArray(d)

        answers = [(0, 3), (4, 7), (8, 11), (12, 15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])

        answers = 4 * [(0, 15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])

    def test_global_limits_cyclic(self):
        """Find the boundaries of a cyclic distribution"""
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), dist=('c', 'n'))
        a = LocalArray(d)
        answers = [(0, 12), (1, 13), (2, 14), (3, 15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])
        answers = 4 * [(0, 15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])

    def test_bad_global_limits(self):
        """ Test that invalid global_limits fails as expected. """
        d = Distribution.from_shape(comm=self.comm, shape=(4, 4))
        a = LocalArray(d)
        with self.assertRaises(InvalidDimensionError):
            a.global_limits(-1)


class TestRankCoords(ParallelTestCase):
    """ Test the rank <--> coords methods. """

    def round_trip(self, la, rank):
        """ Test that given a rank, we can get the coords,
        and then get back to the same rank. """
        coords = la.coords_from_rank(rank)
        # I am not sure what to expect for specific values for coords.
        # Therefore the specific return value is not checked.
        rank2 = la.rank_from_coords(coords)
        self.assertEqual(rank, rank2)

    def test_rank_coords(self):
        """ Test that we can go from rank to coords and back. """
        d = Distribution.from_shape(comm=self.comm, shape=(4, 4))
        la = LocalArray(d)
        max_size = self.comm.Get_size()
        for rank in range(max_size):
            self.round_trip(la, rank=rank)


class TestArrayConversion(ParallelTestCase):
    """ Test array conversion methods. """

    def setUp(self):
        # On Python3, an 'int' gets converted to 'np.int64' on copy,
        # so we force the numpy type to start with so we get back
        # the same thing.
        self.int_type = np.int64
        self.distribution = Distribution.from_shape(comm=self.comm,
                                                    shape=(4,))
        self.int_larr = LocalArray(self.distribution, dtype=self.int_type)
        self.int_larr.fill(3)

    def test_astype(self):
        """ Test that astype() works as expected. """
        # Convert int array to float.
        float_larr = self.int_larr.astype(float)
        for global_inds, value in ndenumerate(float_larr):
            self.assertEqual(value, 3.0)
            self.assertTrue(isinstance(value, float))
        # No type specification for a copy.
        # Should get same type as we started with.
        int_larr2 = self.int_larr.astype(None)
        for global_inds, value in ndenumerate(int_larr2):
            self.assertEqual(value, 3)
            self.assertTrue(isinstance(value, self.int_type))


class TestIndexing(ParallelTestCase):

    def test_indexing_0(self):
        """Can we get and set local elements for a simple dist?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(16, 16), dist=('b', 'n'))
        a = LocalArray(distribution)
        b = LocalArray(distribution)
        for global_inds, value in ndenumerate(a):
            a.global_index[global_inds] = 0.0
        for global_inds, value in ndenumerate(a):
            b.global_index[global_inds] = a.global_index[global_inds]
        for i, value in ndenumerate(a):
            self.assertEqual(b.global_index[i], a.global_index[i])
            self.assertEqual(a.global_index[i], 0.0)

    def test_indexing_1(self):
        """Can we get and set local elements for a complex dist?"""
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(16, 16, 2), dist=('c', 'b', 'n'))
        a = LocalArray(distribution)
        b = LocalArray(distribution)
        for i, value in ndenumerate(a):
            a.global_index[i] = 0.0
        for i, value in ndenumerate(a):
            b.global_index[i] = a.global_index[i]
        for i, value in ndenumerate(a):
            self.assertEqual(b.global_index[i], a.global_index[i])
            self.assertEqual(a.global_index[i], 0.0)

    def test_pack_unpack_index(self):
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(16, 16, 2), dist=('c', 'b', 'n'))
        a = LocalArray(distribution)
        for global_inds, value in ndenumerate(a):
            packed_ind = a.pack_index(global_inds)
            self.assertEqual(global_inds, a.unpack_index(packed_ind))


class TestSlicing(ParallelTestCase):

    comm_size = 2

    def test_slicing(self):
        distribution = Distribution.from_shape(self.comm,
                                               (16, 16),
                                               dist=('b', 'n'))
        a = ones(distribution)
        if self.comm.Get_rank() == 0:
            dd00 = {"dist_type": 'b',
                    "size": 5,
                    "start": 0,
                    "stop": 3,
                    "proc_grid_size": 2,
                    "proc_grid_rank": 0}
            dd01 = {"dist_type": 'n',
                    "size": 16}

            new_distribution = Distribution(self.comm, [dd00, dd01])
            rvals = a.global_index.get_slice((slice(5, None), slice(None)),
                                             new_distribution=new_distribution)
            assert_array_equal(rvals, np.ones((3, 16)))

        elif self.comm.Get_rank() == 1:
            dd10 = {"dist_type": 'b',
                    "size": 5,
                    "start": 3,
                    "stop": 5,
                    "proc_grid_size": 2,
                    "proc_grid_rank": 1}
            dd11 = {"dist_type": 'n',
                    "size": 16}
            new_distribution = Distribution(self.comm, [dd10, dd11])
            rvals = a.global_index.get_slice((slice(None, 10), slice(None)),
                                             new_distribution=new_distribution)
            assert_array_equal(rvals, np.ones((2, 16)))

class TestLocalArrayMethods(ParallelTestCase):

    ddpr = [
        ({'dist_type': 'c',
          'block_size': 1,
          'size': 4,
          'start': 0,
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
         },
         {'dist_type': 'c',
          'block_size': 2,
          'size': 8,
          'start': 0,
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
         }),

        ({'dist_type': 'c',
          'block_size': 1,
          'size': 4,
          'start': 0,
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
         },
         {'dist_type': 'c',
          'block_size': 2,
          'size': 8,
          'start': 2,
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
         }),

        ({'dist_type': 'c',
          'block_size': 1,
          'size': 4,
          'start': 1,
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
         },
         {'dist_type': 'c',
          'block_size': 2,
          'size': 8,
          'start': 0,
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
         }),

        ({'dist_type': 'c',
          'block_size': 1,
          'size': 4,
          'start': 1,
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
         },
         {'dist_type': 'c',
          'block_size': 2,
          'size': 8,
          'start': 2,
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
         })
         ]

    def test_copy_bn(self):
        distribution = Distribution.from_shape(comm=self.comm,
                                        shape=(16, 16), dist=('b', 'n'))
        a = LocalArray(distribution, dtype=np.int_)
        a.fill(11)
        b = a.copy()
        assert_localarrays_equal(a, b, check_dtype=True)

    def test_copy_cbc(self):
        distribution = Distribution(comm=self.comm, dim_data=self.ddpr[self.comm.Get_rank()])
        a = LocalArray(distribution, dtype=np.int_)
        a.fill(12)
        b = a.copy()
        assert_localarrays_equal(a, b, check_dtype=True)

    def test_astype_bn(self):
        new_dtype = np.float32
        d = Distribution.from_shape(comm=self.comm,
                             shape=(16, 16), dist=('b', 'n'))
        a = LocalArray(d, dtype=np.int_)
        a.fill(11)
        b = a.astype(new_dtype)
        assert_localarrays_allclose(a, b, check_dtype=False)
        self.assertEqual(b.dtype, new_dtype)
        self.assertEqual(b.ndarray.dtype, new_dtype)

    def test_astype_cbc(self):
        new_dtype = np.int8
        d = Distribution(comm=self.comm, dim_data=self.ddpr[self.comm.Get_rank()])
        a = LocalArray(d, dtype=np.int32)
        a.fill(12)
        b = a.astype(new_dtype)
        assert_localarrays_allclose(a, b, check_dtype=False)
        self.assertEqual(b.dtype, new_dtype)
        self.assertEqual(b.ndarray.dtype, new_dtype)

    def test_asdist_like(self):
        """Test asdist_like for success and failure."""
        d = Distribution.from_shape(comm=self.comm,
                             shape=(16, 16), dist=('b', 'n'))
        a = LocalArray(d)
        b = LocalArray(d)
        new_a = a.asdist_like(b)
        self.assertEqual(id(a), id(new_a))

        d2 = Distribution.from_shape(comm=self.comm,
                              shape=(16, 16), dist=('n', 'b'))
        a = LocalArray(d)
        b = LocalArray(d2)
        self.assertRaises(IncompatibleArrayError, a.asdist_like, b)


class TestComm(ParallelTestCase):

    def test_create_localarray(self):
        # regression test for issue #144
        dist = Distribution.from_shape(comm=self.comm,
                                shape=(16, 16), dist=('n', 'b'))
        la = LocalArray(dist)


class TestNDEnumerate(ParallelTestCase):
    """Make sure we generate indices compatible with __getitem__."""

    def test_ndenumerate(self):
        d = Distribution.from_shape(comm=self.comm,
                             shape=(16, 16, 2), dist=('c', 'b', 'n'))
        a = LocalArray(d)
        for global_inds, value in ndenumerate(a):
            a.global_index[global_inds] = 0.0


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
