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
            self.larr = da.LocalArray((16,16),
                                     grid_shape=(4,),
                                     comm=comm, buf=None, offset=0)

    def test_has_export(self):
        self.assertTrue(hasattr(self.larr, '__distarray__'))

    def test_export_keys(self):
        required_keys = set(("buffer", "dimdata"))
        export_data = self.larr.__distarray__()
        exported_keys = set(export_data.keys())
        self.assertEqual(required_keys, exported_keys)

    def test_export_buffer(self):
        """See if we actually export a buffer."""
        export_data = self.larr.__distarray__()
        memoryview(export_data['buffer'])

    def test_export_dimdata_len(self):
        """Test if there is a `dimdict` for every dimension."""
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        self.assertEqual(len(dimdata), self.larr.ndim)

    def test_export_dimdata_keys(self):
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        required_keys = {"disttype", "periodic", "datasize", "gridrank",
                "gridsize", "indices", "blocksize", "padding"}
        for dimdict in dimdata:
            self.assertEqual(required_keys, dimdict.keys())

    def test_export_dimdata_values(self):
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        valid_disttypes = {None, 'b', 'c', 'bc', 'bp', 'u'}
        for dd in dimdata:
            self.assertIn(dd['disttype'], valid_disttypes)
            self.assertIsInstance(dd['periodic'], bool)
            self.assertIsInstance(dd['datasize'], int)
            self.assertIsInstance(dd['gridrank'], int)
            self.assertIsInstance(dd['gridsize'], int)
            self.assertIsInstance(dd['indices'], slice)
            self.assertIsInstance(dd['blocksize'], int)
            self.assertEqual(len(dd['padding']), 2)


    @unittest.skip("Import not yet implemented.")
    def test_round_trip(self):
        new_larr = da.fromdap(self.larr)
        self.assertIs(new_larr.local_array, self.larr.local_array)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
