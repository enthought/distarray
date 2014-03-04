"""
Tests for distarray.client

Many of these tests require a 4-engine cluster to be running locally.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from six.moves import range
from IPython.parallel import Client
from distarray.client import Context, DistArray
from distarray.testing import IpclusterTestCase


class TestContextCreation(IpclusterTestCase):

    """Test Context Creation"""

    def test_create_Context(self):
        """Can we create a plain vanilla context?"""
        dac = Context(self.dv)
        self.assertIs(dac.view, self.dv)

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        dac = Context(self.dv, targets=[0, 1])
        self.assertIs(dac.view, self.dv)

    def test_create_Context_with_sub_view(self):
        """Context's view must encompass all ranks in the MPI communicator."""
        subview = self.client[:1]
        if not set(subview.targets) < set(self.dv.targets):
            msg = 'Must set up a cluster with at least 2 engines running.'
            raise unittest.SkipTest(msg)
        with self.assertRaises(ValueError):
            Context(subview)

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        targets = [3, 2]
        dac = Context(self.dv, targets=targets)
        self.assertEqual(set(dac.targets), set(dac.target_to_rank.keys()))
        self.assertEqual(set(range(len(dac.targets))),
                         set(dac.target_to_rank.values()))


class TestDistArray(IpclusterTestCase):

    def more_setUp(self):
        self.dac = Context(self.dv)

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
        vals = np.random.random(20)
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
        ndarr = np.arange(9).reshape(3, 3)
        for (i, j), val in np.ndenumerate(ndarr):
            dap[i, j] = ndarr[i, j]
        np.testing.assert_array_equal(dap.tondarray(), ndarr)


class TestDistArrayCreation(IpclusterTestCase):

    """Test distarray creation methods"""

    def more_setUp(self):
        self.context = Context(self.dv)

    def test_zeros(self):
        shape = (16, 16)
        zero_distarray = self.context.zeros(shape)
        zero_ndarray = np.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_ones(self):
        shape = (16, 16)
        one_distarray = self.context.ones(shape)
        one_ndarray = np.ones(shape)
        assert_array_equal(one_distarray.tondarray(), one_ndarray)

    def test_empty(self):
        shape = (16, 16)
        empty_distarray = self.context.empty(shape)
        self.assertEqual(empty_distarray.shape, shape)

    def test_fromndarray(self):
        ndarr = np.arange(16).reshape(4, 4)
        distarr = self.context.fromndarray(ndarr)
        for (i, j), val in np.ndenumerate(ndarr):
            self.assertEqual(distarr[i, j], ndarr[i, j])


if __name__ == '__main__':
    unittest.main(verbosity=2)
