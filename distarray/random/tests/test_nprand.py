import unittest

import distarray as da
from distarray.core.error import NullCommError
from distarray.mpi.error import InvalidCommSizeError
from distarray.mpi.mpibase import create_comm_of_size
from distarray.core import maps


class TestBasic(unittest.TestCase):
    """
    Run basic shape/size tests on functions in `nprand.py`.
    """

    def shape_asserts(self, la):
        self.assertEqual(la.shape, (16, 16))
        self.assertEqual(la.dist, ('b', None))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.base_comm, self.comm)
        self.assertEqual(la.comm_size, 4)
        self.assertTrue(la.comm_rank in range(4))
        self.assertEqual(la.ndistdim, 1)
        self.assertEqual(la.distdims, (0,))
        self.assertEqual(la.map_classes, (maps.BlockMap,))
        self.assertEqual(la.comm.Get_topo(), (list(la.grid_shape),[0],[la.comm_rank]))
        self.assertEqual(len(la.maps), 1)
        self.assertEqual(la.maps[0].local_shape, 4)
        self.assertEqual(la.maps[0].shape, 16)
        self.assertEqual(la.maps[0].grid_shape, 4)
        self.assertEqual(la.local_shape, (4, 16))
        self.assertEqual(la.local_array.shape, la.local_shape)
        self.assertEqual(la.local_array.dtype, la.dtype)

    def setUp(self):
        try:
            self.comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")

    def test_beta(self):
        try:
            la = da.beta(2, 5, size=(16, 16), grid_shape=(4,), comm=self.comm)
        except NullCommError:
            pass
        else:
            self.shape_asserts(la)

    def test_normal(self):
        try:
            la = da.normal(size=(16, 16), grid_shape=(4,), comm=self.comm)
        except NullCommError:
            pass
        else:
            self.shape_asserts(la)

    def test_rand(self):
        try:
            la = da.rand(size=(16, 16), grid_shape=(4,), comm=self.comm)
        except NullCommError:
            pass
        else:
            self.shape_asserts(la)

    def test_randint(self):
        try:
            la = da.randint(0, 10, size=(16, 16), grid_shape=(4,),
                            comm=self.comm)
        except NullCommError:
            pass
        else:
            self.shape_asserts(la)

    def test_randn(self):
        try:
            la = da.randn((16, 16), grid_shape=(4,), comm=self.comm)
        except NullCommError:
            pass
        else:
            self.shape_asserts(la)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
