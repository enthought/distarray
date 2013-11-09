import unittest

import distarray as da
from distarray.local import maps

from distarray.testing import MpiTestCase, comm_null_passes


class TestBasic(MpiTestCase):
    """Run basic shape/size tests on functions in `nprand.py`."""

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

    @comm_null_passes
    def test_beta(self):
        la = da.beta(2, 5, size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_normal(self):
        la = da.normal(size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_rand(self):
        la = da.rand(size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_randint(self):
        la = da.randint(0, 10, size=(16, 16), grid_shape=(4,),
                        comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_randn(self):
        la = da.randn((16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
