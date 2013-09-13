import unittest
import numpy as np
from IPython.parallel import Client
from distarray.client import DistArrayContext


class TestDistArrayContext(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]

    def test_create_DAC(self):
        '''Can we create a plain vanilla context?'''
        dac = DistArrayContext(self.dv)
        self.assertIs(dac.view, self.dv)

    def test_create_DAC_with_targets(self):
        '''Can we create a context with a subset of engines?'''
        dac = DistArrayContext(self.dv, targets=[0, 1])
        self.assertIs(dac.view, self.dv)


class TestDistArrayProxy(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        self.dac = DistArrayContext(self.dv)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
