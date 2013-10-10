import unittest
import distarray as da
from numpy.testing import assert_array_equal
from distarray.mpi.mpibase import (MPI, create_comm_of_size,
                                   InvalidCommSizeError)
from distarray.utils import comm_null_passes


class BaseDAPCase(object):
    """Base class for Distributed Array Protocol test cases."""

    def get_comm_size(self):
        return 4

    def setUp(self):
        try:
            self.comm = create_comm_of_size(self.get_comm_size())
        except InvalidCommSizeError:
            msg = "Must run with comm size >= {}."
            raise unittest.SkipTest(msg.format(self.get_comm_size()))
        else:
            if self.comm != MPI.COMM_NULL:
                self.larr = self.get_array()

    @comm_null_passes
    def test_has_export(self):
        self.assertTrue(hasattr(self.larr, '__distarray__'))

    @comm_null_passes
    def test_export_keys(self):
        required_keys = set(("buffer", "dimdata"))
        export_data = self.larr.__distarray__()
        exported_keys = set(export_data.keys())
        self.assertEqual(required_keys, exported_keys)

    @comm_null_passes
    def test_export_buffer(self):
        """See if we actually export a buffer."""
        export_data = self.larr.__distarray__()
        memoryview(export_data['buffer'])

    @comm_null_passes
    def test_export_dimdata_len(self):
        """Test if there is a `dimdict` for every dimension."""
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        self.assertEqual(len(dimdata), self.larr.ndim)

    @comm_null_passes
    def test_export_dimdata_keys(self):
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        required_keys = {"disttype", "periodic", "datasize", "gridrank",
                "gridsize", "indices", "blocksize", "padding"}
        for dimdict in dimdata:
            self.assertEqual(required_keys, set(dimdict.keys()))

    @comm_null_passes
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

    @comm_null_passes
    def test_round_trip_equality(self):
        larr = da.fromdap(self.larr, comm=self.comm)
        self.assertEqual(larr.shape, self.larr.shape)
        self.assertEqual(larr.dist, self.larr.dist)
        self.assertEqual(larr.grid_shape, self.larr.grid_shape)
        self.assertEqual(larr.comm_size, self.larr.comm_size)
        self.assertEqual(larr.ndistdim, self.larr.ndistdim)
        self.assertEqual(larr.distdims, self.larr.distdims)
        self.assertEqual(larr.map_classes, self.larr.map_classes)
        self.assertEqual(larr.comm.Get_topo(), self.larr.comm.Get_topo())
        self.assertEqual(len(larr.maps), len(self.larr.maps))
        self.assertEqual(larr.maps[0].local_shape,
                         self.larr.maps[0].local_shape)
        self.assertEqual(larr.maps[0].shape, self.larr.maps[0].shape)
        self.assertEqual(larr.maps[0].grid_shape, self.larr.maps[0].grid_shape)
        self.assertEqual(larr.local_shape, self.larr.local_shape)
        self.assertEqual(larr.local_array.shape, self.larr.local_array.shape)
        self.assertEqual(larr.local_array.dtype, self.larr.local_array.dtype)
        assert_array_equal(larr.local_array, self.larr.local_array)

    @comm_null_passes
    def test_round_trip_identity(self):
        larr = da.fromdap(self.larr, comm=self.comm)
        larr.local_array[0,0] = 99
        assert_array_equal(larr.local_array, self.larr.local_array)
        #self.assertIs(larr.local_array.data, self.larr.local_array.data)


class TestDAPBasic(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((16,16), grid_shape=(4,), comm=self.comm,
                             buf=None, offset=0)


class TestDAPExplicitNone0(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((16,16),
                             dist={0: 'b', 1: None},
                             grid_shape=(4,),
                             comm=self.comm)


class TestDAPExplicitNone1(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((30,60),
                             dist={0: None, 1: 'b'},
                             grid_shape=(4,),
                             comm=self.comm)


class TestDAPTwoDistDims(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((53,77),
                             dist={0: 'b', 1: 'b'},
                             grid_shape=(2, 2),
                             comm=self.comm)


class TestDAPThreeBlockDims(BaseDAPCase, unittest.TestCase):

    def get_array(self):
        return da.LocalArray((53,77,99),
                             dist={0: 'b', 1: 'b', 2: 'b'},
                             grid_shape=(2, 2, 3),
                             comm=self.comm)

    def get_comm_size(self):
        return 12


class TestDAPThreeMixedDims(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((53,77,99),
                             dist={0: 'b', 1: None, 2: 'b'},
                             grid_shape=(2, 2),
                             comm=self.comm)


@unittest.skip("DAP not yet supported for cyclic distribution.")
class TestDAPCyclicDim(BaseDAPCase, unittest.TestCase):
    def get_array(self):
        return da.LocalArray((53,77),
                             dist={0: 'c', 1: 'b'},
                             grid_shape=(2, 2),
                             comm=self.comm)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
