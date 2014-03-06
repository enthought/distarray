"""
Tests for distarray.client

Many of these tests require a 4-engine cluster to be running locally.
"""

import unittest
import numpy
from numpy.testing import assert_array_equal
from six.moves import range
from random import shuffle
from IPython.parallel import Client
from distarray.client import DistArray
from distarray.context import Context


class TestContextCreation(unittest.TestCase):
    """Test Context Creation"""

    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.dv = cls.client[:]
        if len(cls.dv.targets) < 4:
            errmsg = 'Must set up a cluster with at least 4 engines running.'
            raise unittest.SkipTest(errmsg)

    @classmethod
    def tearDownClass(cls):
        """Close the client connections"""
        cls.client.close()

    def tearDown(self):
        """Clear the namespace on the engines after each test."""
        self.dv.clear()

    def test_create_Context(self):
        """Can we create a plain vanilla context?"""
        dac = Context(self.client)
        self.assertIs(dac.client, self.client)

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        dac = Context(self.client, targets=[0, 1])
        self.assertIs(dac.client, self.client)

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        targets = [3, 2]
        dac = Context(self.client, targets=targets)
        self.assertEqual(set(dac.targets), set(dac.target_to_rank.keys()))
        self.assertEqual(set(range(len(dac.targets))),
                         set(dac.target_to_rank.values()))

    def test_context_target_reordering(self):
        '''Are contexts' targets reordered in a consistent way?'''
        orig_targets = self.dv.targets
        ctx1 = Context(self.client, targets=shuffle(orig_targets[:]))
        ctx2 = Context(self.client, targets=shuffle(orig_targets[:]))
        self.assertEqual(ctx1.targets, ctx2.targets)


class TestDistArray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.client = Client()
        self.dv = self.client[:]
        if len(self.dv.targets) < 4:
            errmsg = 'Must set up a cluster with at least 4 engines running.'
            raise unittest.SkipTest(errmsg)
        self.dac = Context(self.client)

    @classmethod
    def tearDownClass(cls):
        cls.client.clear()
        cls.client.close()

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
        dap = self.dac.empty((100,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[111]

    def test_set_index_error(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        with self.assertRaises(IndexError):
            dap[111] = 55

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


class TestDistArrayCreation(unittest.TestCase):
    """Test distarray creation methods"""

    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.dv = cls.client[:]
        if len(cls.dv.targets) < 4:
            errmsg = 'Must set up a cluster with at least 4 engines running.'
            raise unittest.SkipTest(errmsg)
        cls.context = Context(cls.client)

    @classmethod
    def tearDownClass(cls):
        """Clear the namespace and close the client connections after
        this class' tests are run."""
        cls.client.close()

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


class TestReduceMethods(unittest.TestCase):
    """Test reduction methods"""

    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.context = Context(cls.client)

        cls.arr = numpy.arange(16).reshape(4, 4)
        cls.darr = cls.context.fromndarray(cls.arr)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_sum(self):
        np_sum = self.arr.sum()
        da_sum = self.darr.sum()
        self.assertEqual(da_sum, np_sum)

    def test_mean(self):
        np_mean = self.arr.mean()
        da_mean = self.darr.mean()
        self.assertEqual(da_mean, np_mean)

    def test_var(self):
        np_var = self.arr.var()
        da_var = self.darr.var()
        self.assertEqual(da_var, np_var)

    def test_std(self):
        np_std = self.arr.std()
        da_std = self.darr.std()
        self.assertEqual(da_std, np_std)


if __name__ == '__main__':
    unittest.main(verbosity=2)
