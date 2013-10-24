import unittest
import distarray as da
from distarray.mpi.mpibase import create_comm_of_size, InvalidCommSizeError
from distarray.core.error import NullCommError


class TestDistributedArrayProtocol(unittest.TestCase):

    def setUp(self):
        try:
            self.comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest('Must run with comm size > 4.')

    def test_export(self):
        try:
            self.arr = da.LocalArray((16,16),
                                     grid_shape=(4,),
                                     comm=self.comm, buf=None, offset=0)
        except NullCommError:
            pass
        else:
            self.assertIsInstance(self.arr, da.LocalArray)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
