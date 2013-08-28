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

    def test_slice_block_dist(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        vals = np.random.random(20)
        dap[20:40] = vals
        np.testing.assert_allclose(np.array(dap[20:40]), vals)

    def test_slice_set_to_scalar_block_dist(self):
        dap = self.dac.empty((100,), dist={0: 'b'})
        val = 55
        dap[20:40] = val
        np.testing.assert_allclose(np.array(dap[20:40]), val)

    def test_slice_cyclic_dist(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        vals = np.random.random(20)
        dap[20:40] = vals
        np.testing.assert_allclose(np.array(dap[20:40]), vals)

    def test_slice_set_to_scalar_cyclic_dist(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        val = 55
        dap[20:40] = val
        np.testing.assert_allclose(np.array(dap[20:40]), val)

    def test_slice_size_error(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        self.assertRaises(ValueError, dap.__setitem__, slice(20, 40), (11, 12))

    def test_get_index_error(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        self.assertRaises(IndexError, dap.__getitem__, 111)

    def test_set_index_error(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        self.assertRaises(IndexError, dap.__setitem__, 111, 55)

    def test_iteration(self):
        dap = self.dac.empty((100,), dist={0: 'c'})
        vals = range(100)
        dap[:] = vals
        i = 0
        for val in dap:
            self.assertEqual(val, i)
            i += 1


if __name__ == '__main__':
    unittest.main(verbosity=2)
