# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for distarray's client-side API.

Many of these tests require a 4-engine cluster to be running locally.  The
engines should be launched with MPI, using the MPIEngineSetLauncher.

"""

import unittest

import numpy
from numpy.testing import assert_array_equal, assert_allclose

from distarray.externals.six.moves import range
from distarray.testing import DefaultContextTestCase
from distarray.globalapi.distarray import DistArray
from distarray.globalapi.maps import Distribution


class TestDistArray(DefaultContextTestCase):

    def test_set_and_getitem_block_dist(self):
        size = 10
        distribution = Distribution(self.context, (size,), dist={0: 'b'})
        dap = self.context.empty(distribution)

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

        for i in range(1, size + 1):
            dap[-i] = i
            self.assertEqual(dap[-i], i)

    def test_set_and_getitem_nd_block_dist(self):
        size = 5
        distribution = Distribution(self.context, (size, size),
                                    dist={0: 'b', 1: 'b'})
        dap = self.context.empty(distribution)

        for row in range(size):
            for col in range(size):
                val = size*row + col
                dap[row, col] = val
                self.assertEqual(dap[row, col], val)

        for row in range(1, size + 1):
            for col in range(1, size + 1):
                dap[-row, -col] = row + col
                self.assertEqual(dap[-row, -col], row + col)

    def test_set_and_getitem_cyclic_dist(self):
        size = 10
        distribution = Distribution(self.context, (size,),
                                    dist={0: 'c'})
        dap = self.context.empty(distribution)

        for val in range(size):
            dap[val] = val
            self.assertEqual(dap[val], val)

        for i in range(1, size + 1):
            dap[-i] = i
            self.assertEqual(dap[-i], i)

    def test_get_index_error(self):
        distribution = Distribution(self.context, (10,), dist={0: 'c'})
        dap = self.context.empty(distribution)
        with self.assertRaises(IndexError):
            dap[11]
        with self.assertRaises(IndexError):
            dap[-11]

    def test_set_index_error(self):
        distribution = Distribution(self.context, (10,), dist={0: 'c'})
        dap = self.context.empty(distribution)
        with self.assertRaises(IndexError):
            dap[11] = 55
        with self.assertRaises(IndexError):
            dap[-11] = 55

    def test_iteration(self):
        size = 10
        distribution = Distribution(self.context, (size,), dist={0: 'c'})
        dap = self.context.empty(distribution)
        dap.fill(10)
        for val in dap:
            self.assertEqual(val, 10)

    def test_tondarray(self):
        distribution = Distribution(self.context, (3, 3))
        dap = self.context.empty(distribution)
        ndarr = numpy.arange(9).reshape(3, 3)
        for (i, j), val in numpy.ndenumerate(ndarr):
            dap[i, j] = ndarr[i, j]
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)

    def test__array_interface(self):
        distribution = Distribution(self.context, (7, 9))
        darr = self.context.ones(distribution)
        nparr = numpy.array(darr)
        assert_array_equal(darr.tondarray(), nparr)
        assert_array_equal(numpy.ones((7, 9)), nparr)

    def test_set_numpy_array_full_slice_with_distarray(self):
        distribution = Distribution(self.context, (7, 9))
        darr = self.context.ones(distribution)
        nparr = numpy.zeros_like(darr)
        nparr[...] = darr
        assert_array_equal(nparr, darr.toarray())

    def test_set_numpy_array_partial_slice_with_distarray(self):
        distribution = Distribution(self.context, (15, 4))
        darr = self.context.ones(distribution)
        nparr = numpy.zeros_like(darr)
        nparr[3, :] = darr[3, :]
        assert_array_equal(nparr[3, :], darr[3, :].toarray())

    def test_set_numpy_array_partial_slice_with_distarray_2(self):
        distribution = Distribution(self.context, (13, 17))
        darr = self.context.ones(distribution)
        nparr = numpy.zeros((20, 20))
        nparr[:13, :17] = darr
        assert_array_equal(nparr[:13, :17], darr.toarray())

    def test_global_tolocal_bug(self):
        # gh-issue #154
        distribution = Distribution(self.context, (3, 3), dist=('n', 'b'))
        dap = self.context.zeros(distribution)
        ndarr = numpy.zeros((3, 3))
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)


class TestGetItemSlicing(DefaultContextTestCase):

    def test_full_slice_block_dist(self):
        size = 10
        expected = numpy.random.randint(11, size=size)
        arr = self.context.fromarray(expected)
        out = arr[:]
        self.assertSequenceEqual(out.dist, ('b',))
        assert_array_equal(out.toarray(), expected)

    def test_partial_slice_block_dist(self):
        size = 10
        expected = numpy.random.randint(10, size=size)
        arr = self.context.fromarray(expected)
        out = arr[0:2]
        self.assertSequenceEqual(out.dist, ('b',))
        assert_array_equal(out.toarray(), expected[0:2])

    def test_slice_a_slice_block_dist_0(self):
        size = 10
        expected = numpy.random.randint(10, size=size)
        arr = self.context.fromarray(expected)
        s0 = arr[:9]
        s1 = s0[0:5]
        s2 = s1[:2]
        self.assertSequenceEqual(s2.dist, ('b',))
        assert_array_equal(s2.toarray(), expected[:2])

    def test_slice_a_slice_block_dist_1(self):
        size = 10
        expected = numpy.random.randint(10, size=size)
        arr = self.context.fromarray(expected)
        s0 = arr[:9]
        s1 = s0[0:5]
        s2 = s1[-2:]
        self.assertSequenceEqual(s2.dist, ('b',))
        assert_array_equal(s2.toarray(), expected[3:5])

    def test_slice_block_dist_1d_with_step(self):
        size = 10
        step = 2
        expected = numpy.random.randint(10, size=size)
        darr = self.context.fromarray(expected)
        out = darr[::step]
        self.assertSequenceEqual(out.dist, ('b',))
        assert_array_equal(out.toarray(), expected[::step])

    def test_partial_slice_block_dist_2d(self):
        shape = (10, 20)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[2:6, 3:10]
        self.assertSequenceEqual(out.dist, ('b', 'n'))
        assert_array_equal(out.toarray(), expected[2:6, 3:10])

    def test_partial_negative_slice_block_dist_2d(self):
        shape = (10, 20)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[-6:-2, -10:-3]
        self.assertSequenceEqual(out.dist, ('b', 'n'))
        assert_array_equal(out.toarray(),
                           expected[-6:-2, -10:-3])

    def test_incomplete_slice_block_dist_2d(self):
        shape = (10, 20)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[3:9]
        self.assertSequenceEqual(out.dist, ('b', 'n'))
        assert_array_equal(out.toarray(), expected[3:9])

    def test_incomplete_index_block_dist_2d(self):
        shape = (10, 20)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[1]
        self.assertSequenceEqual(out.dist, ('n'))
        assert_array_equal(out.toarray(), expected[1])

    def test_empty_slice_1d(self):
        shape = (10,)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[100:]
        self.assertEqual(out.shape, (0,))
        self.assertEqual(out.grid_shape, (1,))
        self.assertEqual(len(out.targets), 1)
        assert_array_equal(out.toarray(), expected[100:])

    def test_empty_slice_2d(self):
        shape = (10, 20)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[100:, 100:]
        self.assertEqual(out.shape, (0, 0))
        self.assertEqual(out.grid_shape, (1, 1))
        self.assertEqual(len(out.targets), 1)
        assert_array_equal(out.toarray(), expected[100:, 100:])

    def test_trailing_ellipsis(self):
        shape = (2, 3, 7, 6)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[1, ...]
        self.assertSequenceEqual(out.dist, ('n', 'n', 'n'))
        assert_array_equal(out.toarray(), expected[1, ...])

    def test_leading_ellipsis(self):
        shape = (2, 3, 7, 6)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        out = arr[..., 3]
        self.assertSequenceEqual(out.dist, ('b', 'n', 'n'))
        assert_array_equal(out.toarray(), expected[..., 3])

    def test_multiple_ellipsis(self):
        shape = (2, 4, 2, 4, 1, 5)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        assert_array_equal(arr[..., 3, ..., 4].toarray(),
                           expected[..., 3, ..., 4])

    def test_vestigial_ellipsis(self):
        shape = (1, 2, 3)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        assert_array_equal(arr[0, :, 0, ...].toarray(),
                           expected[0, :, 0, ...])

    def test_all_ellipsis(self):
        shape = (3, 2, 4)
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        assert_array_equal(arr[..., ..., ..., ...].toarray(),
                           expected[..., ..., ..., ...])

    def test_resulting_slice(self):
        dist = Distribution(self.context, (10, 20))
        da = self.context.ones(dist)
        db = da[:5, :10]
        dc = db * 2
        assert_array_equal(dc.toarray(), numpy.ones(dc.shape) * 2)

    def test_0d_empty_tuple(self):
        shape = ()
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        assert_array_equal(arr[()], expected[()])

    def test_0d_ellipsis(self):
        shape = ()
        expected = numpy.random.randint(10, size=shape)
        arr = self.context.fromarray(expected)
        assert_array_equal(arr[...].toarray(), expected[...])


class TestSetItemSlicing(DefaultContextTestCase):

    def test_small_1d_slice(self):
        source = numpy.random.randint(10, size=20)
        new_data = numpy.random.randint(10, size=3)
        slc = slice(1, 4)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_large_1d_slice(self):
        source = numpy.random.randint(10, size=20)
        new_data = numpy.random.randint(10, size=10)
        slc = slice(5, 15)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_1d_slice_with_step(self):
        source = numpy.random.randint(10, size=20)
        new_data = numpy.random.randint(10, size=5)
        slc = slice(7, 17, 2)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_2d_slice_0(self):
        # on process boundaries
        source = numpy.random.randint(10, size=(10, 20))
        new_data = numpy.random.randint(10, size=(5, 10))
        slc = (slice(5, 10), slice(5, 15))
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_2d_slice_with_step(self):
        source = numpy.random.randint(10, size=(10, 20))
        new_data = numpy.random.randint(10, size=(2, 5))
        slc = (slice(5, 10, 3), slice(5, 15, 2))
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_2d_slice_1(self):
        # off process boundaries
        source = numpy.random.randint(10, size=(10, 20))
        new_data = numpy.random.randint(10, size=(5, 10))
        slc = (slice(3, 8), slice(9, 19))
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_full_3d_slice(self):
        source = numpy.random.randint(10, size=(3, 4, 5))
        new_data = numpy.random.randint(10, size=(3, 4, 5))
        slc = (slice(None), slice(None), slice(None))
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_full_3d_slice_ellipsis(self):
        source = numpy.random.randint(10, size=(3, 4, 5))
        new_data = numpy.random.randint(10, size=(3, 4, 5))
        slc = Ellipsis
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_3d_slice_ellipsis_with_step(self):
        source = numpy.random.randint(10, size=(5, 4, 5))
        new_data = numpy.random.randint(10, size=(5, 2, 5))
        slc = (Ellipsis, slice(None, None, 2), Ellipsis)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_partial_indexing_0(self):
        source = numpy.random.randint(10, size=(3, 4, 5))
        new_data = numpy.random.randint(10, size=(4, 5))
        slc = (1,)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_partial_indexing_1(self):
        source = numpy.random.randint(10, size=(3, 4, 5))
        new_data = numpy.random.randint(10, size=(3, 5))
        slc = (slice(None), 2)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_non_array_data(self):
        source = numpy.random.randint(10, size=(3, 4))
        new_data = [42, 42, 42, 42]
        slc = (2,)
        arr = self.context.fromarray(source)
        source[slc] = new_data
        arr[slc] = new_data
        assert_array_equal(arr.toarray(), source)

    def test_valueerror(self):
        source = numpy.random.randint(10, size=21)
        new_data = numpy.random.randint(10, size=10)
        slc = slice(15, None)
        arr = self.context.fromarray(source)
        with self.assertRaises(ValueError):
            arr[slc] = new_data

    def test_set_DistArray_slice(self):
        dist = Distribution(self.context, (10, 20))
        da = self.context.ones(dist)
        db = self.context.zeros(dist)
        da[...] = db

    def test_set_0d_slice(self):
        expected = numpy.array(33)
        arr = self.context.fromarray(expected)

        val0 = 55
        arr[...] = val0
        assert_array_equal(arr[...].toarray(), numpy.array(val0))

        val1 = 99
        arr[()] = val1
        assert_array_equal(arr[...].toarray(), numpy.array(val1))


class TestDistArrayCreationFromGlobalDimData(DefaultContextTestCase):

    def test_from_global_dim_data_irregular_block(self):

        bounds = (0, 2, 3, 4, 10)
        glb_dim_data = (
                {'dist_type': 'b',
                 'bounds': bounds},
                )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         glb_dim_data)
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_from_global_dim_data_1d(self):
        total_size = 40
        list_of_indices = [
                [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
                [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
                [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
                [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
                ]
        glb_dim_data = (
                {'dist_type': 'u',
                    'indices': list_of_indices,
                    },
                )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         glb_dim_data)
        distarr = DistArray(distribution, dtype=int)
        for i in range(total_size):
            distarr[i] = i
        localarrays = distarr.get_localarrays()
        for i, arr in enumerate(localarrays):
            assert_allclose(arr, list_of_indices[i])

    def test_from_global_dim_data_bu(self):

        rows = 9
        row_break_point = rows // 2
        cols = 10
        col_indices = numpy.random.permutation(range(cols))
        col_break_point = len(col_indices) // 3
        indices = [col_indices[:col_break_point], col_indices[col_break_point:]]
        glb_dim_data = (
                {
                    'dist_type': 'b',
                    'bounds': (0, row_break_point, rows)
                },
                {
                    'dist_type': 'u',
                    'indices' : indices
                },
            )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         glb_dim_data)
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_from_global_dim_data_bc(self):
        """ Test creation of a block-cyclic array. """

        rows, cols = 5, 9
        global_dim_data = (
                # dim 0
                {
                    'dist_type': 'c',
                    'proc_grid_size': 2,
                    'size': rows,
                    'block_size': 2,
                },
                # dim 1
                {
                    'dist_type': 'c',
                    'proc_grid_size': 2,
                    'size': cols,
                    'block_size': 2,
                },)
        distribution = Distribution.from_global_dim_data(self.context,
                                                         global_dim_data)
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()
        las = distarr.get_localarrays()
        local_shapes = [la.local_shape for la in las]
        self.assertSequenceEqual(local_shapes,
                                 [(3, 5), (3, 4), (2, 5), (2, 4)])

    def test_from_global_dim_data_uu(self):
        rows = 6
        cols = 20
        row_ixs = numpy.random.permutation(range(rows))
        col_ixs = numpy.random.permutation(range(cols))
        row_indices = [row_ixs[:rows//2], row_ixs[rows//2:]]
        col_indices = [col_ixs[:cols//4], col_ixs[cols//4:]]
        glb_dim_data = (
                {'dist_type': 'u',
                    'indices': row_indices},
                {'dist_type': 'u',
                    'indices' : col_indices},
                )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         glb_dim_data)
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_global_dim_data_local_dim_data_equivalence(self):
        rows, cols = 5, 9
        glb_dim_data = (
                {'dist_type': 'c',
                 'block_size': 2,
                 'size': rows,
                 'proc_grid_size': 2,
                 },
                {'dist_type': 'c',
                 'block_size': 2,
                 'proc_grid_size': 2,
                 'size': cols,
                 },
                )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         glb_dim_data)
        actual = distribution.get_dim_data_per_rank()

        expected = [
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': rows,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': cols,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': rows,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': cols,
              'start': 2}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': rows,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': cols,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': rows,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': cols,
              'start': 2}),
        ]
        self.assertSequenceEqual(actual, expected)

    def test_irregular_block_assignment(self):
        global_dim_data = (
                {
                    'dist_type': 'b',
                    'bounds': (0, 5),
                },
                {
                    'dist_type': 'b',
                    'bounds': (0, 2, 6, 7, 9),
                }
            )
        distribution = Distribution.from_global_dim_data(self.context,
                                                         global_dim_data)
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()


class TestDistArrayCreation(DefaultContextTestCase):

    """Test distarray creation methods"""

    def test___init__(self):
        shape = (5, 5)
        distribution = Distribution(self.context, shape, ('b', 'c'))
        da = DistArray(distribution, dtype=int)
        da.fill(42)
        nda = numpy.empty(shape, dtype=int)
        nda.fill(42)
        assert_array_equal(da.tondarray(), nda)

    def test_zeros(self):
        shape = (16, 16)
        distribution = Distribution(self.context, shape)
        zero_distarray = self.context.zeros(distribution)
        zero_ndarray = numpy.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_zeros_0d(self):
        shape = ()
        distribution = Distribution(self.context, shape)
        zero_distarray = self.context.zeros(distribution)
        zero_ndarray = numpy.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_ones(self):
        shape = (16, 16)
        distribution = Distribution(self.context, shape)
        ones_distarray = self.context.ones(distribution)
        ones_ndarray = numpy.ones(shape)
        assert_array_equal(ones_distarray.tondarray(), ones_ndarray)

    def test_ones_0d(self):
        shape = ()
        distribution = Distribution(self.context, shape)
        ones_distarray = self.context.ones(distribution)
        ones_ndarray = numpy.ones(shape)
        assert_array_equal(ones_distarray.tondarray(), ones_ndarray)

    def test_empty(self):
        shape = (16, 16)
        distribution = Distribution(self.context, shape)
        empty_distarray = self.context.empty(distribution)
        self.assertEqual(empty_distarray.shape, distribution.shape)

    def test_empty_0d(self):
        shape = ()
        distribution = Distribution(self.context, shape)
        empty_distarray = self.context.empty(distribution)
        self.assertEqual(empty_distarray.shape, distribution.shape)

    def test_fromndarray(self):
        ndarr = numpy.arange(16).reshape(4, 4)
        distarr = self.context.fromndarray(ndarr)
        for (i, j), val in numpy.ndenumerate(ndarr):
            self.assertEqual(distarr[i, j], ndarr[i, j])

    def test_fromndarray_0d(self):
        ndarr = numpy.array(42)
        distarr = self.context.fromarray(ndarr)
        assert_array_equal(ndarr, distarr.toarray())

    def test_grid_rank(self):
        # regression test for issue #235
        d = Distribution(self.context, (4, 4, 4),
                                    dist=('b', 'n', 'b'),
                                    grid_shape=(1, 1, 4))
        a = self.context.empty(d)
        self.assertEqual(a.grid_shape, (1, 1, 4))

    def test_fromfunction(self):
        def fn(i, j):
            return i + j
        shape = (7, 9)
        expected = numpy.fromfunction(fn, shape, dtype=int)
        result = self.context.fromfunction(fn, shape, dtype=int)
        assert_array_equal(expected, result.tondarray())


class TestDistArrayCreationSubSet(DefaultContextTestCase):

    def test_create_target_subset(self):
        shape = (100, 100)
        subtargets = self.context.targets[::2]
        distribution = Distribution(self.context, shape=shape,
                                    targets=subtargets)
        darr = self.context.ones(distribution)
        lss = darr.localshapes()
        self.assertEqual(len(lss), len(subtargets))

        ddpr = distribution.get_dim_data_per_rank()
        self.assertEqual(len(ddpr), len(subtargets))


class TestReductionRegression(DefaultContextTestCase):
    ''' Separate class necessary b/c need to run on at least 9 engines to exercise
    regression.

    '''

    ntargets = 9

    def test_reduction(self):
        """ Tests that GH issue 403 is fixed """
        arr = numpy.arange(9 * 9).reshape(9, 9)
        dist = Distribution(self.context, arr.shape, ('b', 'b'), (9, 1))
        darr = self.context.fromndarray(arr, dist)
        assert_allclose(darr.sum().tondarray(), arr.sum())


class TestReduceMethods(DefaultContextTestCase):
    """Test reduction methods"""

    @classmethod
    def setUpClass(cls):
        super(TestReduceMethods, cls).setUpClass()
        cls.arr = numpy.arange(16).reshape(4, 4)
        dist = Distribution(cls.context, cls.arr.shape, ('b', 'b'), (2, 2))
        cls.darr = cls.context.fromndarray(cls.arr, dist)

    def test_sum_last_axis(self):
        da_sum = self.darr.sum(axis=-1)
        da_sum2 = self.darr.sum(axis=(1,))
        assert_allclose(da_sum.tondarray(), da_sum2.tondarray())

    def test_sum_axis_none(self):
        np_sum = self.arr.sum(axis=None)
        da_sum = self.darr.sum(axis=None)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_multiaxis(self):
        np_sum = self.arr.sum(axis=(0, 1))
        da_sum = self.darr.sum(axis=(0, 1))
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_0d(self):
        arr = numpy.arange(16)
        darr = self.context.fromndarray(arr)
        np_sum = arr.sum(axis=0)
        da_sum = darr.sum(axis=0)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_chained(self):
        np_sum = self.arr.sum(axis=0).sum(axis=0)
        da_sum = self.darr.sum(axis=0).sum(axis=0)
        self.assertEqual(da_sum.ndim, 0)
        self.assertEqual(np_sum.ndim, 0)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_along_axis0(self):
        np_sum = self.arr.sum(axis=0)
        da_sum = self.darr.sum(axis=0)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_along_axis1(self):
        np_sum = self.arr.sum(axis=1)
        da_sum = self.darr.sum(axis=1)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_sum_dtype(self):
        da_sum = self.darr.sum(axis=0, dtype=int)
        np_sum = self.arr.sum(axis=0, dtype=int)
        assert_allclose(da_sum.tondarray(), np_sum)

    def test_mean_axis_none(self):
        np_mean = self.arr.mean(axis=None)
        da_mean = self.darr.mean(axis=None)
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_mean_multiaxis(self):
        np_mean = self.arr.mean(axis=(0, 1))
        da_mean = self.darr.mean(axis=(0, 1))
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_mean_along_axis_0(self):
        da_mean = self.darr.mean(axis=0)
        np_mean = self.arr.mean(axis=0)
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_mean_along_axis_1(self):
        da_mean = self.darr.mean(axis=1)
        np_mean = self.arr.mean(axis=1)
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_mean_dtype(self):
        da_mean = self.darr.mean(axis=0, dtype=int)
        np_mean = self.arr.mean(axis=0, dtype=int)
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_mean_chained(self):
        np_mean = self.arr.mean(axis=0).mean(axis=0)
        da_mean = self.darr.mean(axis=0).mean(axis=0)
        self.assertEqual(da_mean.ndim, 0)
        self.assertEqual(np_mean.ndim, 0)
        assert_allclose(da_mean.tondarray(), np_mean)

    def test_var(self):
        np_var = self.arr.var()
        da_var = self.darr.var()
        self.assertEqual(da_var.tondarray(), np_var)

    def test_var_axis_0(self):
        np_var = self.arr.var(axis=0)
        da_var = self.darr.var(axis=0)
        assert_allclose(da_var.tondarray(), np_var)

    def test_var_axis_1(self):
        np_var = self.arr.var(axis=1)
        da_var = self.darr.var(axis=1)
        assert_allclose(da_var.tondarray(), np_var)

    def test_var_dtype(self):
        np_var = self.arr.var(dtype=int)
        da_var = self.darr.var(dtype=int)
        self.assertEqual(da_var.toarray(), np_var)

    def test_std(self):
        np_std = self.arr.std()
        da_std = self.darr.std()
        self.assertEqual(da_std.toarray(), np_std)

    def test_std_dtype(self):
        np_std = self.arr.std(dtype=int)
        da_std = self.darr.std(dtype=int)
        self.assertEqual(da_std.toarray(), np_std)

    def test_std_axis_0(self):
        np_std = self.arr.std(axis=0)
        da_std = self.darr.std(axis=0)
        assert_allclose(da_std.tondarray(), np_std)

    def test_std_axis_1(self):
        np_std = self.arr.std(axis=1)
        da_std = self.darr.std(axis=1)
        assert_allclose(da_std.tondarray(), np_std)

    def test_min(self):
        np_min = self.arr.min()
        da_min = self.darr.min()
        assert_allclose(da_min.tondarray(), np_min)

    def test_min_axis_1(self):
        np_min = self.arr.min(axis=1)
        da_min = self.darr.min(axis=1)
        assert_allclose(da_min.tondarray(), np_min)

    def test_max(self):
        np_max = self.arr.max()
        da_max = self.darr.max()
        assert_allclose(da_max.tondarray(), np_max)

    def test_max_axis_1(self):
        np_max = self.arr.max(axis=1)
        da_max = self.darr.max(axis=1)
        assert_allclose(da_max.tondarray(), np_max)

    def test_sum_4D_cyclic(self):
        shape = (10, 20, 30, 40)
        arr = numpy.zeros(shape)
        arr.fill(3)
        dist = Distribution(self.context, shape=shape,
                            dist=('c', 'c', 'c', 'c'))
        darr = self.context.empty(shape_or_dist=dist)
        darr.fill(3)
        for axis in range(4):
            arr_sum = arr.sum(axis=axis)
            darr_sum = darr.sum(axis=axis)
            assert_allclose(darr_sum.tondarray(), arr_sum)
        assert_allclose(darr.sum().tondarray(), arr.sum())

    def test_gh_435_regression_with_var(self):
        dist = Distribution(self.context, shape=(14,), dist=('b'),
                            targets=range(4))
        darr = self.context.ones(dist)
        darr.var()

    def test_boolean_reduction(self):
        nparr = numpy.random.randn(200)
        darr = self.context.fromarray(nparr)
        mask = darr < 2
        np_mask = mask.tondarray()
        assert_array_equal(mask.min().tondarray(), np_mask.min())
        assert_array_equal(mask.max().tondarray(), np_mask.max())
        assert_array_equal(mask.sum().tondarray(), np_mask.sum())
        assert_array_equal(mask.mean().tondarray(), np_mask.mean())
        assert_allclose(mask.var().tondarray(), np_mask.var())
        assert_allclose(mask.std().tondarray(), np_mask.std())


class TestFromLocalArrays(DefaultContextTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFromLocalArrays, cls).setUpClass()
        cls.distribution = Distribution((4, 4))
        cls.distarray = cls.context.ones(cls.distribution, dtype=int)
        cls.expected = numpy.ones((4, 4), dtype=int)

    def with_context(self):
        da = DistArray.from_localarrays(self.distarray.key,
                                        context=self.context)
        assert_array_equal(da.toarray(), self.expected)

    def with_context_and_dtype(self):
        da = DistArray.from_localarrays(self.distarray.key,
                                        context=self.context, dtype=int)
        assert_array_equal(da.toarray(), self.expected)

    def with_distribution(self):
        da = DistArray.from_localarrays(self.distarray.key,
                                        distribution=self.distribution)
        assert_array_equal(da.toarray(), self.expected)

    def with_distribution_and_dtype(self):
        da = DistArray.from_localarrays(self.distarray.key,
                                        distribution=self.distribution,
                                        dtype=int)
        assert_array_equal(da.toarray(), self.expected)

    def with_distribution_and_context(self):
        with self.assertRaise(RuntimeError):
            DistArray.from_localarrays(self.distarray.key,
                                       context=self.context,
                                       distribution=self.distribution)


class TestView(DefaultContextTestCase):

    def test_plain_view(self):
        a = numpy.zeros((4, 5), dtype=numpy.float32)
        da = self.context.fromndarray(a)

        da_view = da.view()
        da_view[3, 4] = 99
        assert_array_equal(da_view.tondarray(), da.tondarray())

        da_view = da.view(dtype=da.dtype.str)
        da_view[2, 2] = 33
        assert_array_equal(da_view.tondarray(), da.tondarray())

    def test_smaller_dtype(self):
        a = numpy.asarray(numpy.arange(20).reshape((4, 5)), dtype=numpy.int32)
        da = self.context.fromndarray(a)

        dtype = numpy.int16
        da_view = da.view(dtype=dtype)
        da_view[3, 4] = -55
        assert_array_equal(da_view.tondarray(), da.tondarray().view(dtype))

    def test_larger_dtype(self):
        a = numpy.asarray(numpy.arange(40).reshape((4, 10)), dtype=numpy.int32)
        da = self.context.fromndarray(a)

        dtype = numpy.int16
        da_view = da.view(dtype=dtype)
        da_view[2, 1] = -33
        assert_array_equal(da_view.tondarray(), da.tondarray().view(dtype))

    def test_harder_distribution(self):
        a = numpy.asarray(numpy.arange(30), dtype=numpy.int16)
        gdd = (dict(dist_type='b', bounds=[0, 3, 12, 21, 30]),)
        dist = Distribution.from_global_dim_data(context=self.context,
                                                 global_dim_data=gdd)
        da = self.context.fromndarray(a, distribution=dist)

        dtype = numpy.int32
        with self.assertRaises(ValueError):
            da.view(dtype=dtype)

    def test_incompatible_dimsize(self):
        a = numpy.zeros((4, 5), dtype=numpy.int32)
        da = self.context.fromndarray(a)

        dtype = numpy.int64
        with self.assertRaises(ValueError):
            da.view(dtype=dtype)

    def test_incompatible_dtype(self):
        a = numpy.zeros((4, 5), dtype=numpy.int32)
        da = self.context.fromndarray(a)

        dtype = "S7"
        with self.assertRaises(ValueError):
            da.view(dtype=dtype)


if __name__ == '__main__':
    unittest.main(verbosity=2)
