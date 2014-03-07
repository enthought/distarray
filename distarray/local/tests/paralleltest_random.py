import unittest

from distarray.local import random as local_random
from distarray.local import maps

from distarray.testing import MpiTestCase, comm_null_passes


class TestBasic(MpiTestCase):
    """Run basic shape/size tests on functions in `random.py`."""

    def shape_asserts(self, la):
        self.assertEqual(la.shape, (16, 16))
        self.assertEqual(la.dist, ('b', 'n'))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.base_comm, self.comm)
        self.assertEqual(la.comm_size, 4)
        self.assertTrue(la.comm_rank in range(4))
        self.assertEqual(la.ndistdim, 1)
        self.assertEqual(la.distdims, (0,))
        self.assertEqual(la.comm.Get_topo(),
                         (list(la.grid_shape),
                          [0],[la.comm_rank]))
        self.assertEqual(len(la.maps), 2)
        self.assertEqual(la.shape, (16, 16))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.local_shape, (4, 16))
        self.assertEqual(la.local_array.shape, la.local_shape)
        self.assertEqual(la.local_array.dtype, la.dtype)

    @comm_null_passes
    def test_beta(self):
        la = local_random.beta(2, 5, size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_normal(self):
        la = local_random.normal(size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_rand(self):
        la = local_random.rand(size=(16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_randint(self):
        la = local_random.randint(0, 10, size=(16, 16), grid_shape=(4,),
                        comm=self.comm)
        self.shape_asserts(la)

    @comm_null_passes
    def test_randn(self):
        la = local_random.randn((16, 16), grid_shape=(4,), comm=self.comm)
        self.shape_asserts(la)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
