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
from distarray.client import DistArray
from distarray.client_map import Distribution
from distarray.context import Context


class TestDistArray(unittest.TestCase):

    def setUp(self):
        self.dac = Context()

    def tearDown(self):
        self.dac.close()

    def test_set_and_getitem_block_dist(self):
        size = 10
        dap = self.dac.empty((size,), dist={0: 'b'})

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

    def test_set_and_getitem_nd_block_dist(self):
        size = 5
        dap = self.dac.empty((size, size), dist={0: 'b', 1: 'b'})

        for row in range(size):
            for col in range(size):
                val = size*row + col
                dap[row, col] = val

        for row in range(size):
            for col in range(size):
                val = size*row + col
                self.assertEqual(dap[row, col], val)

    def test_set_and_getitem_cyclic_dist(self):
        size = 10
        dap = self.dac.empty((size,), dist={0: 'c'})

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

    def test_get_index_error(self):
        dap = self.dac.empty((10,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[11]

    def test_set_index_error(self):
        dap = self.dac.empty((10,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[11] = 55

    def test_iteration(self):
        size = 10
        dap = self.dac.empty((size,), dist={0: 'c'})
        dap.fill(10)
        for val in dap:
            self.assertEqual(val, 10)

    def test_tondarray(self):
        dap = self.dac.empty((3, 3))
        ndarr = numpy.arange(9).reshape(3, 3)
        for (i, j), val in numpy.ndenumerate(ndarr):
            dap[i, j] = ndarr[i, j]
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)

    def test_global_tolocal_bug(self):
        # gh-issue #154
        dap = self.dac.zeros((3, 3), dist=('n', 'b'))
        ndarr = numpy.zeros((3, 3))
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)


class TestDistArrayCreationFromGlobalDimData(unittest.TestCase):

    def setUp(self):
        self.context = Context()

    def tearDown(self):
        self.context.close()

    def test_from_global_dim_data_irregular_block(self):
        global_size = 10
        bounds = (0, 2, 3, 4, 10)
        glb_dim_data = (
                {'dist_type': 'b',
                    'bounds': bounds},
                )
        distarr = self.context.from_global_dim_data(glb_dim_data)
        for i in range(global_size):
            distarr[i] = i

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
        distarr = self.context.from_global_dim_data(glb_dim_data)
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
        distarr = self.context.from_global_dim_data(glb_dim_data)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j

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
        distarr = self.context.from_global_dim_data(global_dim_data)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j
        las = distarr.get_localarrays()
        local_shapes = [la.local_shape for la in las]
        self.assertSequenceEqual(local_shapes, [(3,5), (3,4), (2,5), (2,4)])

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
        distarr = self.context.from_global_dim_data(glb_dim_data)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j

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
        mdmap = Distribution(self.context, glb_dim_data)
        actual = mdmap.get_dim_data_per_rank()

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
        global_shape = (5, 9)
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
        distarr = self.context.from_global_dim_data(global_dim_data)
        for i in range(global_shape[0]):
            for j in range(global_shape[1]):
                distarr[i, j] = i + j


class TestDistArrayCreation(unittest.TestCase):

    """Test distarray creation methods"""

    def setUp(self):
        self.context = Context()

    def tearDown(self):
        self.context.close()

    def test___init__(self):
        shape = (100, 100)
        mdmap = Distribution.from_shape(self.context, shape, ('b', 'c'))
        da = DistArray(mdmap, dtype=int)
        da.fill(42)
        nda = numpy.empty(shape, dtype=int)
        nda.fill(42)
        assert_array_equal(da.tondarray(), nda)

    def test_zeros(self):
        shape = (16, 16)
        zero_distarray = self.context.zeros(shape)
        zero_ndarray = numpy.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_ones(self):
        shape = (16, 16)
        one_distarray = self.context.ones(shape)
        one_ndarray = numpy.ones(shape)
        assert_array_equal(one_distarray.tondarray(), one_ndarray)

    def test_empty(self):
        shape = (16, 16)
        empty_distarray = self.context.empty(shape)
        self.assertEqual(empty_distarray.shape, shape)

    def test_fromndarray(self):
        ndarr = numpy.arange(16).reshape(4, 4)
        distarr = self.context.fromndarray(ndarr)
        for (i, j), val in numpy.ndenumerate(ndarr):
            self.assertEqual(distarr[i, j], ndarr[i, j])

    def test_grid_rank(self):
        # regression test for issue #235
        a = self.context.empty((4, 4, 4), dist=('b', 'n', 'b'),
                               grid_shape=(1, 1, 4))
        self.assertEqual(a.grid_shape, (1, 1, 4))


class TestReduceMethods(unittest.TestCase):
    """Test reduction methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.arr = numpy.arange(16).reshape(4, 4)
        cls.darr = cls.context.fromndarray(cls.arr)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()

    def test_sum(self):
        np_sum = self.arr.sum()
        da_sum = self.darr.sum()
        self.assertEqual(da_sum, np_sum)

    def test_sum_dtype(self):
        np_sum = self.arr.sum(dtype=int)
        da_sum = self.darr.sum(dtype=int)
        self.assertEqual(da_sum, np_sum)

    def test_mean(self):
        np_mean = self.arr.mean()
        da_mean = self.darr.mean()
        self.assertEqual(da_mean, np_mean)

    def test_mean_dtype(self):
        np_mean = self.arr.mean(dtype=int)
        da_mean = self.darr.mean(dtype=int)
        self.assertEqual(da_mean, np_mean)

    def test_var(self):
        np_var = self.arr.var()
        da_var = self.darr.var()
        self.assertEqual(da_var, np_var)

    def test_var_dtype(self):
        np_var = self.arr.var(dtype=int)
        da_var = self.darr.var(dtype=int)
        self.assertEqual(da_var, np_var)

    def test_std(self):
        np_std = self.arr.std()
        da_std = self.darr.std()
        self.assertEqual(da_std, np_std)

    def test_std_dtype(self):
        np_std = self.arr.std(dtype=int)
        da_std = self.darr.std(dtype=int)
        self.assertEqual(da_std, np_std)


if __name__ == '__main__':
    unittest.main(verbosity=2)
