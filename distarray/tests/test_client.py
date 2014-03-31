# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

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
from distarray.client_map import ClientMDMap
from distarray.context import Context
from distarray.testing import IpclusterTestCase


class TestDistArray(IpclusterTestCase):

    def setUp(self):
        self.dac = Context(self.client)

     # overloads base class...
    def tearDown(self):
        del self.dac
        super(TestDistArray, self).tearDown()

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

    @unittest.skip("Slicing not yet implemented.")
    def test_slice_in_getitem_block_dist(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        self.assertIsInstance(dap[20:40], DistArray)

    @unittest.skip("Slicing not yet implemented.")
    def test_slice_in_setitem_raises_valueerror(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        vals = numpy.random.random(20)
        with self.assertRaises(NotImplementedError):
            dap[20:40] = vals

    @unittest.skip('Slice assignment not yet implemented.')
    def test_slice_size_error(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        with self.assertRaises(NotImplementedError):
            dap[20:40] = (11, 12)

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


class TestDistArrayCreation(IpclusterTestCase):

    """Test distarray creation methods"""

    def setUp(self):
        self.context = Context(self.client)

     # overloads base class...
    def tearDown(self):
        del self.context
        super(TestDistArrayCreation, self).tearDown()

    def test___init__(self):
        shape = (100, 100)
        mdmap = ClientMDMap(self.context, shape, ('b', 'c'))
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

    def test_from_dim_data_1d(self):
        total_size = 40
        ddpp = [
            ({'dist_type': 'u',
              'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'size': total_size},)]
        distarr = self.context.from_dim_data(ddpp)
        for i in range(total_size):
            distarr[i] = i
        localarrays = distarr.get_localarrays()
        for i, arr in enumerate(localarrays):
            assert_allclose(arr, ddpp[i][0]['indices'])

    def test_from_dim_data_irregular_block(self):
        global_size = 10
        starts = (0, 2, 3, 4)
        stops = (2, 3, 4, 10)
        ddpp = [
             (
              {'dist_type': 'b',
               'start': starts[i],
               'stop': stops[i],
               'proc_grid_rank': i,
               'proc_grid_size': 4,
               'size': global_size},
              ) for i in range(4)
             ]
        distarr = self.context.from_dim_data(ddpp)
        for i in range(global_size):
            distarr[i] = i

    def test_from_dim_data_bu(self):
        rows = 9
        cols = 10
        col_indices = numpy.random.permutation(range(cols))
        row_break_point = rows // 2
        col_break_point = len(col_indices) // 3
        ddpp = [
             (
              {'dist_type': 'b',
               'start': 0,
               'stop': row_break_point,
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:col_break_point],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': 0,
               'stop': row_break_point,
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[col_break_point:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': row_break_point,
               'stop': rows,
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:col_break_point],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': row_break_point,
               'stop': rows,
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[col_break_point:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             )]
        distarr = self.context.from_dim_data(ddpp)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j

    def test_from_dim_data_uu(self):
        rows = 6
        cols = 20
        row_indices = numpy.random.permutation(range(rows))
        col_indices = numpy.random.permutation(range(cols))
        ddpp = [
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows//2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols//4],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows//2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols//4:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows//2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols//4],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows//2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols//4:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             )]
        distarr = self.context.from_dim_data(ddpp)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j


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
        del cls.darr
        del cls.arr
        del cls.context

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
