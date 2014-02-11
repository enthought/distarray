import unittest
import numpy as np

import distarray.remote.denseremotearray as da
from distarray import utils
from distarray.testing import comm_null_passes, MpiTestCase
from distarray.remote.error import IncompatibleArrayError


class TestInit(MpiTestCase):

    """Is the __init__ method working properly?"""

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.RemoteArray((16,16), grid_shape=(4,), comm=self.comm,
                                  buf=None)

    @comm_null_passes
    def test_basic(self):
        """Test basic RemoteArray creation."""
        self.assertEqual(self.larr.shape, (16,16))
        self.assertEqual(self.larr.dist, ('b', None))
        self.assertEqual(self.larr.grid_shape, (4,))
        self.assertEqual(self.larr.base_comm, self.comm)
        self.assertEqual(self.larr.comm_size, 4)
        self.assertTrue(self.larr.comm_rank in range(4))
        self.assertEqual(self.larr.ndistdim, 1)
        self.assertEqual(self.larr.distdims, (0,))
        self.assertEqual(self.larr.comm.Get_topo(),
                         (list(self.larr.grid_shape),
                          [0], [self.larr.comm_rank]))
        self.assertEqual(len(self.larr.maps), 1)
        self.assertEqual(self.larr.grid_shape, (4,))
        self.assertEqual(self.larr.shape, (16, 16))
        self.assertEqual(self.larr.remote_shape, (4, 16))
        self.assertEqual(self.larr.remote_array.shape, self.larr.remote_shape)
        self.assertEqual(self.larr.remote_array.dtype, self.larr.dtype)

    @comm_null_passes
    def test_remotearray(self):
        """Can the remote_array be set and get?"""
        self.larr.get_remotearray()
        la = np.random.random(self.larr.remote_shape)
        la = np.asarray(la, dtype=self.larr.dtype)
        self.larr.set_remotearray(la)
        self.larr.get_remotearray()


class TestFromDimdata(MpiTestCase):

    def assert_alike(self, l0, l1):
        self.assertEqual(l0.shape, l1.shape)
        self.assertEqual(l0.dist, l1.dist)
        self.assertEqual(l0.grid_shape, l1.grid_shape)
        self.assertEqual(l0.base_comm, l1.base_comm)
        self.assertEqual(l0.comm_size, l1.comm_size)
        self.assertEqual(l0.comm_rank, l1.comm_rank)
        self.assertEqual(l0.ndistdim, l1.ndistdim)
        self.assertEqual(l0.distdims, l1.distdims)
        self.assertEqual(l0.comm.Get_topo(), l1.comm.Get_topo())
        self.assertEqual(len(l0.maps), len(l1.maps))
        self.assertEqual(l0.shape, l1.shape)
        self.assertEqual(l0.grid_shape, l1.grid_shape)
        self.assertEqual(l0.remote_shape, l1.remote_shape)
        self.assertEqual(l0.remote_array.shape, l1.remote_array.shape)
        self.assertEqual(l0.remote_array.dtype, l1.remote_array.dtype)
        self.assertEqual(list(l0.maps[0].global_index),
                         list(l1.maps[0].global_index))
        self.assertEqual(list(l0.maps[0].remote_index),
                         list(l1.maps[0].remote_index))

    @comm_null_passes
    def test_block(self):
        dim0 = {
            "disttype": 'b',
            "datasize": 16,
            "gridsize": 4,
            }

        dim1 = {
            "disttype": None,
            "datasize": 16,
            "gridsize": None,
        }

        dimdata = (dim0, dim1)

        larr = da.RemoteArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.RemoteArray((16,16), dist={0: 'b'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic(self):
        dim0 = {
            "disttype": None,
            "datasize": 16,
            "gridsize": None,
            }

        dim1 = {
            "disttype": 'c',
            "datasize": 16,
            "gridsize": 4,
            }

        dimdata = (dim0, dim1)

        larr = da.RemoteArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.RemoteArray((16,16), dist={1: 'c'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic_and_block(self):
        dim0 = {
            "disttype": 'c',
            "datasize": 16,
            "gridsize": 2,
            }

        dim1 = {
            "disttype": 'b',
            "datasize": 16,
            "gridsize": 2,
            }

        dimdata = (dim0, dim1)

        larr = da.RemoteArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.RemoteArray((16,16), dist={0: 'c', 1: 'b'},
                                 grid_shape=(2, 2), comm=self.comm)

        self.assert_alike(larr, expected)

    @unittest.skip('Not implemented.')
    def test_block_cyclic(self):
        dim0 = {"disttype": 'bc',
                "datasize": 16,
                "gridsize": 4,
                "blocksize": 2}

        dim1 = {"disttype": None,
                "datasize": 16,
                "gridsize": None}

        dimdata = (dim0, dim1)

        larr = da.RemoteArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.RemoteArray((16,16), dist={0: 'bc'}, blocksize=2,
                                 grid_shape=(2, 2), comm=self.comm)

        self.assert_alike(larr, expected)


class TestGridShape(MpiTestCase):

    def get_comm_size(self):
        return 12

    @comm_null_passes
    def test_grid_shape(self):
        """Test various ways of setting the grid_shape."""
        self.larr = da.RemoteArray((20,20), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (3,4))
        self.larr = da.RemoteArray((2*10,6*10), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,6))
        self.larr = da.RemoteArray((6*10,2*10), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (6,2))
        self.larr = da.RemoteArray((100,10,300), dist=('b',None,'c'), comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,6))
        self.larr = da.RemoteArray((100,50,300), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,2,3))


class TestDistMatrix(MpiTestCase):

    """Test the dist_matrix."""

    def get_comm_size(self):
        return 12

    @unittest.skip("Plot test.")
    @comm_null_passes
    def test_plot_dist_matrix(self):
        """Can we create and possibly plot a dist_matrix?"""
        la = da.RemoteArray((10,10), dist=('c','c'), comm=self.comm)
        if self.comm.Get_rank() == 0:
            import pylab
            pylab.ion()
            pylab.matshow(la)
            pylab.colorbar()
            pylab.draw()
            pylab.show()


class TestRemoteInd(MpiTestCase):

    """Test the computation of remote indices."""

    @comm_null_passes
    def test_block_simple(self):
        """Can we compute remote indices for a block distribution?"""
        la = da.RemoteArray((4, 4), comm=self.comm)
        self.assertEqual(la.shape, (4, 4))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.remote_shape, (1, 4))
        row_result = [(0, 0), (0, 1), (0, 2), (0, 3)]

        row = la.comm_rank
        calc_row_result = [la.global_to_remote(row, col) for col in
                           range(la.shape[1])]
        self.assertEqual(row_result, calc_row_result)

    @comm_null_passes
    def test_block_complex(self):
        """Can we compute remote indices for a block distribution?"""
        la = da.RemoteArray((8, 2), comm=self.comm)
        self.assertEqual(la.shape, (8, 2))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.remote_shape, (2, 2))
        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (1, 0), (1, 1)]
        elif la.comm_rank == 1:
            gis = [(2, 0), (2, 1), (3, 0), (3, 1)]
        elif la.comm_rank == 2:
            gis = [(4, 0), (4, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 3:
            gis = [(6, 0), (6, 1), (7, 0), (7, 1)]

        result = [la.global_to_remote(*gi) for gi in gis]
        self.assertEqual(result, expected_lis)

    @comm_null_passes
    def test_cyclic_simple(self):
        """Can we compute remote indices for a cyclic distribution?"""
        la = da.RemoteArray((10,), dist={0: 'c'}, comm=self.comm)
        self.assertEqual(la.shape, (10,))
        self.assertEqual(la.grid_shape, (4,))

        if la.comm_rank == 0:
            gis = (0, 4, 8)
            self.assertEqual(la.remote_shape, (3,))
            calc_result = [la.global_to_remote(gi) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 1:
            gis = (1, 5, 9)
            self.assertEqual(la.remote_shape, (3,))
            calc_result = [la.global_to_remote(gi) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 2:
            gis = (2, 6)
            self.assertEqual(la.remote_shape, (2,))
            calc_result = [la.global_to_remote(gi) for gi in gis]
            result = [(0,), (1,)]
        elif la.comm_rank == 3:
            gis = (3, 7)
            self.assertEqual(la.remote_shape, (2,))
            calc_result = [la.global_to_remote(gi) for gi in gis]
            result = [(0,), (1,)]

        self.assertEqual(result, calc_result)

    @comm_null_passes
    def test_cyclic_complex(self):
        """Can we compute remote indices for a cyclic distribution?"""
        la = da.RemoteArray((8, 2), dist={0: 'c'}, comm=self.comm)
        self.assertEqual(la.shape, (8, 2))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.remote_shape, (2, 2))

        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (4, 0), (4, 1)]
        elif la.comm_rank == 1:
            gis = [(1, 0), (1, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 2:
            gis = [(2, 0), (2, 1), (6, 0), (6, 1)]
        elif la.comm_rank == 3:
            gis = [(3, 0), (3, 1), (7, 0), (7, 1)]

        result = [la.global_to_remote(*gi) for gi in gis]
        self.assertEqual(result, expected_lis)


class TestGlobalInd(MpiTestCase):

    """Test the computation of global indices."""

    def round_trip(self, la):
        for indices in utils.multi_for([range(s) for s in la.remote_shape]):
            gi = la.remote_to_global(*indices)
            li = la.global_to_remote(*gi)
            self.assertEqual(li,indices)

    @comm_null_passes
    def test_block(self):
        """Can we go from global to remote indices and back for block?"""
        la = da.RemoteArray((4,4), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_cyclic(self):
        """Can we go from global to remote indices and back for cyclic?"""
        la = da.RemoteArray((8,8), dist=('c',None), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_crazy(self):
        """Can we go from global to remote indices and back for a complex case?"""
        la = da.RemoteArray((10,100,20), dist=('b','c',None), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_global_limits_block(self):
        """Find the boundaries of a block distribution"""
        a = da.RemoteArray((16, 16), dist=('b', None), comm=self.comm)

        answers = [(0, 3), (4, 7), (8, 11), (12, 15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])

        answers = 4 * [(0, 15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])

    @comm_null_passes
    def test_global_limits_cyclic(self):
        """Find the boundaries of a cyclic distribution"""
        a = da.RemoteArray((16,16), dist=('c',None), comm=self.comm)
        answers = [(0,12),(1,13),(2,14),(3,15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])
        answers = 4*[(0,15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])


class TestIndexing(MpiTestCase):

    @comm_null_passes
    def test_indexing_0(self):
        """Can we get and set remote elements for a simple dist?"""
        a = da.RemoteArray((16,16), dist=('b',None), comm=self.comm)
        b = da.RemoteArray((16,16), dist=('b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_indexing_1(self):
        """Can we get and set remote elements for a complex dist?"""
        a = da.RemoteArray((16,16,2), dist=('c','b',None), comm=self.comm)
        b = da.RemoteArray((16,16,2), dist=('c','b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_pack_unpack_index(self):
        a = da.RemoteArray((16,16,2), dist=('c','b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            packed_ind = a.pack_index(global_inds)
            self.assertEqual(global_inds, a.unpack_index(packed_ind))


class TestRemoteArrayMethods(MpiTestCase):

    @comm_null_passes
    def test_asdist_like(self):
        """Test asdist_like for success and failure."""
        a = da.RemoteArray((16,16), dist=('b',None), comm=self.comm)
        b = da.RemoteArray((16,16), dist=('b',None), comm=self.comm)
        new_a = a.asdist_like(b)
        self.assertEqual(id(a),id(new_a))
        a = da.RemoteArray((16,16), dist=('b',None), comm=self.comm)
        b = da.RemoteArray((16,16), dist=(None,'b'), comm=self.comm)
        self.assertRaises(IncompatibleArrayError, a.asdist_like, b)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass

