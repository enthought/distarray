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

    def test_has_export(self):
        self.assertTrue(hasattr(self.arr, '__distarray__'))

    def test_well_formedness(self):
        required_keys = set(("buffer", "dimdata"))
        export = self.arr.__distarray__()
        exported_keys = set(export.keys())
        self.assertEqual(required_keys, exported_keys)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
