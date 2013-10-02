import unittest
import distarray as da
from distarray.mpi.mpibase import create_comm_of_size, InvalidCommSizeError


class TestDistributedArrayProtocol(unittest.TestCase):

    def setUp(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest('Must run with comm size > 4.')
        else:
            self.arr = da.LocalArray((16,16),
                                     grid_shape=(4,),
                                     comm=comm, buf=None, offset=0)

    def test_export(self):
        self.assertIsInstance(self.arr, da.LocalArray)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
