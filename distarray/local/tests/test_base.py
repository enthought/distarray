import unittest

from six.moves import range
from distarray.remote.base import BaseRemoteArray


class TestBaseRemoteArray(unittest.TestCase):

    def test_block(self):
        dim0 = {
            "disttype": 'b',
            "datasize": 16,
            "gridsize": 1,
            "gridrank": 0,
            "start": 0,
            "stop": 16,
            }

        dim1 = {
            "disttype": None,
            "datasize": 16,
            }

        dimdata = (dim0, dim1)

        larr = BaseRemoteArray(dimdata)

        self.assertEqual(larr.grid_shape, (1,))
        self.assertEqual(larr.shape, (16, 16))
        self.assertEqual(len(larr.maps), 1)
        self.assertEqual(larr.remote_array.shape, larr.shape)
        self.assertEqual(larr.ndim, 2)
        self.assertEqual(larr.size, 16*16)
        self.assertEqual(larr.dist, ('b', None))
        self.assertEqual(larr.distdims, (0,))
        self.assertEqual(larr.ndistdim, 1)
        self.assertEqual(larr.remote_size, 16*16)

        self.assertEqual([x for x in larr.maps[0].global_index],
                         [x for x in range(16)])

    def test_cyclic(self):
        dim0 = {
            "disttype": 'c',
            "datasize": 16,
            "gridsize": 1,
            "gridrank": 0,
            "start": 0,
            }

        dim1 = {
            "disttype": 'b',
            "datasize": 16,
            "gridsize": 1,
            "gridrank": 0,
            "start": 0,
            "stop": 16,
            }

        dimdata = (dim0, dim1)

        larr = BaseRemoteArray(dimdata)

        self.assertEqual(larr.grid_shape, (1, 1))
        self.assertEqual(larr.shape, (16, 16))
        self.assertEqual(len(larr.maps), 2)
        self.assertEqual(larr.remote_array.shape, larr.shape)
        self.assertEqual(larr.ndim, 2)
        self.assertEqual(larr.size, 16*16)
        self.assertEqual(larr.dist, ('c', 'b'))
        self.assertEqual(larr.distdims, (0, 1))
        self.assertEqual(larr.ndistdim, 2)
        self.assertEqual(larr.remote_size, 16*16)

        self.assertEqual([x for x in larr.maps[0].global_index],
                         [x for x in range(0, 16, 1)])
