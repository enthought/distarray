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
from numpy.testing import assert_array_equal

from distarray.externals.six.moves import range
from distarray.client import DistArray
from distarray.client_map import ClientMDMap
from distarray.context import Context
from distarray.creation import empty, zeros, fromndarray
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
        dap = empty((size,), dist={0: 'b'})

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

    def test_set_and_getitem_nd_block_dist(self):
        size = 5
        dap = empty((size, size), dist={0: 'b', 1: 'b'})

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
        dap = empty((size,), dist={0: 'c'})

        for val in range(size):
            dap[val] = val

        for val in range(size):
            self.assertEqual(dap[val], val)

    @unittest.skip("Slicing not yet implemented.")
    def test_slice_in_getitem_block_dist(self):
        dap = empty((100,), dist={0: 'b'})
        self.assertIsInstance(dap[20:40], DistArray)

    @unittest.skip("Slicing not yet implemented.")
    def test_slice_in_setitem_raises_valueerror(self):
        dap = empty((100,), dist={0: 'b'})
        vals = numpy.random.random(20)
        with self.assertRaises(NotImplementedError):
            dap[20:40] = vals

    @unittest.skip('Slice assignment not yet implemented.')
    def test_slice_size_error(self):
        dap = empty((100,), dist={0: 'c'})
        with self.assertRaises(NotImplementedError):
            dap[20:40] = (11, 12)

    def test_get_index_error(self):
        dap = empty((10,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[11]

    def test_set_index_error(self):
        dap = empty((10,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[11] = 55

    def test_iteration(self):
        size = 10
        dap = empty((size,), dist={0: 'c'})
        dap.fill(10)
        for val in dap:
            self.assertEqual(val, 10)

    def test_tondarray(self):
        dap = empty((3, 3))
        ndarr = numpy.arange(9).reshape(3, 3)
        for (i, j), val in numpy.ndenumerate(ndarr):
            dap[i, j] = ndarr[i, j]
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)

    def test_global_tolocal_bug(self):
        # gh-issue #154
        dap = zeros((3, 3), dist=('n', 'b'))
        ndarr = numpy.zeros((3, 3))
        numpy.testing.assert_array_equal(dap.tondarray(), ndarr)


class TestReduceMethods(unittest.TestCase):
    """Test reduction methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()

        cls.arr = numpy.arange(16).reshape(4, 4)
        cls.darr = fromndarray(cls.arr)

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
