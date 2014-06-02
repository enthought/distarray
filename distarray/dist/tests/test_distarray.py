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
from distarray.dist.distarray import DistArray
from distarray.dist.maps import Distribution
from distarray.dist.context import Context


class TestDistArray(unittest.TestCase):

    def setUp(self):
        self.dac = Context()

    def tearDown(self):
        self.dac.close()

    def test_set_and_getitem_block_dist(self):
        size = 10
        distribution = Distribution.from_shape(self.dac, (size,),
                                               dist={0: 'b'})
        dap = self.dac.empty(distribution)

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

        for i in range(1, size + 1):
            dap[-i] = i
            self.assertEqual(dap[-i], i)

    def test_set_and_getitem_nd_block_dist(self):
        size = 5
        distribution = Distribution.from_shape(self.dac, (size, size),
                                               dist={0: 'b', 1: 'b'})
        dap = self.dac.empty(distribution)

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
        distribution = Distribution.from_shape(self.dac, (size,),
                                               dist={0: 'c'})
        dap = self.dac.empty(distribution)

        for val in range(size):
            dap[val] = val
            self.assertEqual(dap[val], val)

        for i in range(1, size + 1):
            dap[-i] = i
            self.assertEqual(dap[-i], i)

    def test_get_index_error(self):
        distribution = Distribution.from_shape(self.dac, (10,), dist={0: 'c'})
        dap = self.dac.empty(distribution)
        with self.assertRaises(IndexError):
            dap[11]
        with self.assertRaises(IndexError):
            dap[-11]

    def test_set_index_error(self):
        distribution = Distribution.from_shape(self.dac, (10,), dist={0: 'c'})
        dap = self.dac.empty(distribution)
        with self.assertRaises(IndexError):
            dap[11] = 55
        with self.assertRaises(IndexError):
            dap[-11] = 55

    def test_iteration(self):
        size = 10
        distribution = Distribution.from_shape(self.dac, (size,),
                                               dist={0: 'c'})
        dap = self.dac.empty(distribution)
        dap.fill(10)
        for val in dap:
            self.assertEqual(val, 10)

    def test_tondarray(self):
        distribution = Distribution.from_shape(self.dac, (3, 3))
        dap = self.dac.empty(distribution)
        ndarr = numpy.arange(9).reshape(3, 3)
        for (i, j), val in numpy.ndenumerate(ndarr):
            dap[i, j] = ndarr[i, j]
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)

    def test_global_tolocal_bug(self):
        # gh-issue #154
        distribution = Distribution.from_shape(self.dac, (3, 3),
                                               dist=('n', 'b'))
        dap = self.dac.zeros(distribution)
        ndarr = numpy.zeros((3, 3))
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)


class TestDistArrayCreationFromGlobalDimData(unittest.TestCase):

    def setUp(self):
        self.context = Context()

    def tearDown(self):
        self.context.close()

    def test_from_global_dim_data_irregular_block(self):

        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")

        bounds = (0, 2, 3, 4, 10)
        glb_dim_data = (
                {'dist_type': 'b',
                 'bounds': bounds},
                )
        distribution = Distribution(self.context,
                                    glb_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_from_global_dim_data_1d(self):
        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
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
        distribution = Distribution(self.context,
                                    glb_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        for i in range(total_size):
            distarr[i] = i
        localarrays = distarr.get_localarrays()
        for i, arr in enumerate(localarrays):
            assert_allclose(arr, list_of_indices[i])

    def test_from_global_dim_data_bu(self):

        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")

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
        distribution = Distribution(self.context,
                                    glb_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_from_global_dim_data_bc(self):
        """ Test creation of a block-cyclic array. """

        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")

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
        distribution = Distribution(self.context,
                                    global_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()
        las = distarr.get_localarrays()
        local_shapes = [la.local_shape for la in las]
        self.assertSequenceEqual(local_shapes,
                                 [(3, 5), (3, 4), (2, 5), (2, 4)])

    def test_from_global_dim_data_uu(self):
        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
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
        distribution = Distribution(self.context,
                                    glb_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()

    def test_global_dim_data_local_dim_data_equivalence(self):
        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
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
        distribution = Distribution(self.context,
                                    glb_dim_data,
                                    targets=self.context.targets[:4])
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
        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
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
        distribution = Distribution(self.context,
                                    global_dim_data,
                                    targets=self.context.targets[:4])
        distarr = DistArray(distribution, dtype=int)
        distarr.toarray()


class TestDistArrayCreation(unittest.TestCase):

    """Test distarray creation methods"""

    def setUp(self):
        self.context = Context()

    def tearDown(self):
        self.context.close()

    def test___init__(self):
        shape = (5, 5)
        distribution = Distribution.from_shape(self.context, shape, ('b', 'c'))
        da = DistArray(distribution, dtype=int)
        da.fill(42)
        nda = numpy.empty(shape, dtype=int)
        nda.fill(42)
        assert_array_equal(da.tondarray(), nda)

    def test_zeros(self):
        shape = (16, 16)
        distribution = Distribution.from_shape(self.context, shape)
        zero_distarray = self.context.zeros(distribution)
        zero_ndarray = numpy.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_ones(self):
        shape = (16, 16)
        distribution = Distribution.from_shape(self.context, shape)
        one_distarray = self.context.ones(distribution)
        one_ndarray = numpy.ones(shape)
        assert_array_equal(one_distarray.tondarray(), one_ndarray)

    def test_empty(self):
        distribution = Distribution.from_shape(self.context, (16, 16))
        empty_distarray = self.context.empty(distribution)
        self.assertEqual(empty_distarray.shape, distribution.shape)

    def test_fromndarray(self):
        ndarr = numpy.arange(16).reshape(4, 4)
        distarr = self.context.fromndarray(ndarr)
        for (i, j), val in numpy.ndenumerate(ndarr):
            self.assertEqual(distarr[i, j], ndarr[i, j])

    def test_grid_rank(self):
        # regression test for issue #235
        if len(self.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
        d = Distribution.from_shape(self.context, (4, 4, 4),
                                    dist=('b', 'n', 'b'),
                                    grid_shape=(1, 1, 4),
                                    targets=self.context.targets[:4])
        a = self.context.empty(d)
        self.assertEqual(a.grid_shape, (1, 1, 4))

    def test_fromfunction(self):
        fn = lambda i, j: i + j
        shape = (7, 9)
        expected = numpy.fromfunction(fn, shape, dtype=int)
        result = self.context.fromfunction(fn, shape, dtype=int)
        assert_array_equal(expected, result.tondarray())

class TestDistArrayCreationSubSet(unittest.TestCase):

    def setUp(self):
        self.context = Context()

    def tearDown(self):
        self.context.close()

    def test_create_target_subset(self):
        shape = (100, 100)
        subtargets = self.context.targets[::2]
        distribution = Distribution.from_shape(self.context, shape=shape, targets=subtargets)
        darr = self.context.ones(distribution)
        lss = darr.get_localshapes()
        self.assertEqual(len(lss), len(subtargets))

        ddpr = distribution.get_dim_data_per_rank()
        self.assertEqual(len(ddpr), len(subtargets))


class TestReduceMethods(unittest.TestCase):
    """Test reduction methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        if len(cls.context.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
        cls.arr = numpy.arange(16).reshape(4, 4)
        dist = Distribution.from_shape(cls.context,
                                       cls.arr.shape, ('b', 'b'), (2, 2),
                                       targets=cls.context.targets[:4])
        cls.darr = cls.context.fromndarray(cls.arr, dist)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()

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

    def test_mean_along_axis_1(self):
        da_mean = self.darr.mean(axis=0)
        np_mean = self.arr.mean(axis=0)
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
        self.assertEqual(da_var, np_var)

    def test_std(self):
        np_std = self.arr.std()
        da_std = self.darr.std()
        self.assertEqual(da_std, np_std)

    def test_std_dtype(self):
        np_std = self.arr.std(dtype=int)
        da_std = self.darr.std(dtype=int)
        self.assertEqual(da_std, np_std)

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
        dist = Distribution.from_shape(self.context,
                                       shape=shape,
                                       dist=('c', 'c', 'c', 'c'))
        darr = self.context.empty(distribution=dist)
        darr.fill(3)
        for axis in range(4):
            arr_sum = arr.sum(axis=axis)
            darr_sum = darr.sum(axis=axis)
            assert_allclose(darr_sum.tondarray(), arr_sum)
        assert_allclose(darr.sum().tondarray(), arr.sum())


class TestFromLocalArrays(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.distribution = Distribution.from_shape((4, 4))
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

if __name__ == '__main__':
    unittest.main(verbosity=2)
