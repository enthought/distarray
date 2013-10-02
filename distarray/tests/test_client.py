"""
Tests for distarray.client

Many of these tests require a 4-engine cluster to be running locally.
"""

import unittest
import numpy as np
from IPython.parallel import Client
from distarray.client import Context

from six.moves import range as xrange


class TestContext(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        if len(self.dv.targets) < 4:
            raise unittest.SkipTest('Must set up a cluster with at least 4 engines running.')

    def test_create_Context(self):
        '''Can we create a plain vanilla context?'''
        dac = Context(self.dv)
        self.assertIs(dac.view, self.dv)

    def test_create_Context_with_targets(self):
        '''Can we create a context with a subset of engines?'''
        dac = Context(self.dv, targets=[0,1])
        self.assertIs(dac.view, self.dv)

    def test_create_Context_with_sub_view(self):
        '''Context's view must encompass all ranks in the MPI communicator.'''
        subview = self.client[:1]
        if not set(subview.targets) < set(self.dv.targets):
            raise unittest.SkipTest('Must set up a cluster with at least 2 engines running.')
        with self.assertRaises(ValueError):
            Context(subview)

    def test_create_Context_with_targets_ranks(self):
        '''Check that the target <=> rank mapping is consistent.'''
        targets = [3,2]
        dac = Context(self.dv, targets=targets)
        self.assertEqual(set(dac.targets), set(dac.target_to_rank.keys()))
        self.assertEqual(set(range(len(dac.targets))), set(dac.target_to_rank.values()))


class TestDistArray(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        self.dac = Context(self.dv)

    def test_set_and_getitem_block_dist(self):
        dap = self.dac.empty((100,), dist={0: 'b'})

        for val in xrange(100):
            dap[val] = val

        for val in xrange(100):
            self.assertEqual(dap[val], val)

    def test_set_and_getitem_cyclic_dist(self):
        dap = self.dac.empty((100,), dist={0: 'c'})

        for val in xrange(100):
            dap[val] = val

        for val in xrange(100):
            self.assertEqual(dap[val], val)

    def test_slice_in_getitem_raises_valueerror(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        with self.assertRaises(NotImplementedError):
            dap[20:40]

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
        dap = self.dac.empty((100,), dist={0: 'c'})
        dap.fill(10)
        for val in dap:
            self.assertEqual(val, 10)

    def test_owner_rank(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        self.assertEqual(dap.owner_rank(10), 0)
        self.assertEqual(dap.owner_rank(30), 1)
        self.assertEqual(dap.owner_rank(60), 2)
        self.assertEqual(dap.owner_rank(80), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
