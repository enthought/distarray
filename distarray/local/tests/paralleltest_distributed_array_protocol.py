import unittest
import numpy as np
import distarray as da
from distutils.version import StrictVersion
from numpy.testing import assert_array_equal
from distarray.testing import comm_null_passes, MpiTestCase


VALID_DISTTYPES = {None, 'b', 'c', 'bc', 'bp', 'u'}


class DapTestMixin(object):

    """Base test class for DAP test cases.

    You must overload `more_setUp` and add a `self.larr` RemoteArray to
    test.
    """

    @comm_null_passes
    def test_has_export(self):
        self.assertTrue(hasattr(self.larr, '__distarray__'))

    @comm_null_passes
    def test_export_keys(self):
        required_keys = set(("__version__", "buffer", "dimdata"))
        export_data = self.larr.__distarray__()
        exported_keys = set(export_data.keys())
        self.assertEqual(required_keys, exported_keys)

    @comm_null_passes
    def test_export_buffer(self):
        """See if we actually export a buffer."""
        export_data = self.larr.__distarray__()
        memoryview(export_data['buffer'])

    @comm_null_passes
    def test_export_version(self):
        """Check type of version."""
        export_data = self.larr.__distarray__()
        StrictVersion(export_data['__version__'])

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
        required_keys = {"disttype", "datasize"}
        for dimdict in dimdata:
            self.assertTrue(required_keys <= set(dimdict.keys()))

    @comm_null_passes
    def test_export_dimdata_values(self):
        export_data = self.larr.__distarray__()
        dimdata = export_data['dimdata']
        for dd in dimdata:
            self.assertIn(dd['disttype'], VALID_DISTTYPES)
            self.assertIsInstance(dd['datasize'], int)

            for key in ('gridrank', 'gridsize',  'blocksize', 'padding'):
                try:
                    self.assertIsInstance(dd[key], int)
                except KeyError:
                    pass
            try:
                self.assertIsInstance(dd['periodic'], bool)
            except KeyError:
                pass

    @comm_null_passes
    def test_round_trip_equality(self):
        larr = da.RemoteArray.from_distarray(self.larr, comm=self.comm)
        self.assertEqual(larr.shape, self.larr.shape)
        self.assertEqual(larr.dist, self.larr.dist)
        self.assertEqual(larr.grid_shape, self.larr.grid_shape)
        self.assertEqual(larr.comm_size, self.larr.comm_size)
        self.assertEqual(larr.ndistdim, self.larr.ndistdim)
        self.assertEqual(larr.distdims, self.larr.distdims)
        self.assertEqual(larr.comm.Get_topo(), self.larr.comm.Get_topo())
        self.assertEqual(len(larr.maps), len(self.larr.maps))
        self.assertEqual(larr.remote_shape, self.larr.remote_shape)
        self.assertEqual(larr.remote_array.shape, self.larr.remote_array.shape)
        self.assertEqual(larr.remote_array.dtype, self.larr.remote_array.dtype)
        assert_array_equal(larr.remote_array, self.larr.remote_array)

    @comm_null_passes
    def test_round_trip_identity(self):
        larr = da.RemoteArray.from_distarray(self.larr, comm=self.comm)
        if self.comm.Get_rank() == 0:
            idx = (0,) * larr.remote_array.ndim
            larr.remote_array[idx] = 99
        assert_array_equal(larr.remote_array, self.larr.remote_array)
        #self.assertIs(larr.remote_array.data, self.larr.remote_array.data)


class TestDapBasic(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((16, 16), grid_shape=(4,), comm=self.comm)


class TestDapUint(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((16, 16), dtype='uint8', grid_shape=(4,),
                                  comm=self.comm, buf=None)


class TestDapComplex(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((16, 16), dtype='complex128',
                                  grid_shape=(4,), comm=self.comm, buf=None)


class TestDapExplicitNone0(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((16, 16), dist={0: 'b', 1: None},
                                  grid_shape=(4,), comm=self.comm)


class TestDapExplicitNone1(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((30, 60), dist={0: None, 1: 'b'},
                                  grid_shape=(4,), comm=self.comm)


class TestDapTwoDistDims(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((53, 77), dist={0: 'b', 1: 'b'},
                                  grid_shape=(2, 2), comm=self.comm)


class TestDapThreeBlockDims(DapTestMixin, MpiTestCase):

    def get_comm_size(self):
        return 12

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((53, 77, 99),
                                  dist={0: 'b', 1: 'b', 2: 'b'},
                                  grid_shape=(2, 2, 3),
                                  comm=self.comm)


class TestDapCyclicDim(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((53, 77),
                                  dist={0: 'c'},
                                  grid_shape=(4,),
                                  comm=self.comm)


class TestDapCyclicBlock(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((53, 77),
                                  dist={0: 'c', 1: 'b'},
                                  grid_shape=(2, 2),
                                  comm=self.comm)


class TestDapThreeMixedDims(DapTestMixin, MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((53, 77, 99), dtype='float64',
                                  dist={0: 'b', 1: None, 2: 'c'},
                                  grid_shape=(2, 2),
                                  comm=self.comm)


class TestDapLopsided(DapTestMixin, MpiTestCase):

    def get_comm_size(self):
        return 2

    @comm_null_passes
    def more_setUp(self):
        if self.comm.Get_rank() == 0:
            arr = np.arange(20)
        elif self.comm.Get_rank() == 1:
            arr = np.arange(30)

        self.larr = da.RemoteArray((50,), dtype='float64',
                             dist={0: 'b', 1: None},
                             grid_shape=(2,), comm=self.comm, buf=arr)

    @comm_null_passes
    def test_values(self):
        if self.comm.Get_rank() == 0:
            assert_array_equal(np.arange(20), self.larr.remote_array)
        elif self.comm.Get_rank() == 1:
            assert_array_equal(np.arange(30), self.larr.remote_array)

        larr = da.RemoteArray.from_distarray(self.larr, comm=self.comm)
        if self.comm.Get_rank() == 0:
            assert_array_equal(np.arange(20), larr.remote_array)
        elif self.comm.Get_rank() == 1:
            assert_array_equal(np.arange(30), larr.remote_array)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
