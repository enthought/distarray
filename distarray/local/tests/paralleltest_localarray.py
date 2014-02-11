import unittest
import numpy as np

import distarray.local.denselocalarray as da
from distarray import utils
from distarray.testing import comm_null_passes, MpiTestCase
from distarray.local.error import IncompatibleArrayError


class TestInit(MpiTestCase):

    """Is the __init__ method working properly?"""

    @comm_null_passes
    def more_setUp(self):
        self.larr_1d = da.LocalArray((7,), grid_shape=(4,), comm=self.comm,
                                     buf=None)
        self.larr_2d = da.LocalArray((16,16), grid_shape=(4,), comm=self.comm,
                                     buf=None)

    @comm_null_passes
    def test_basic_1d(self):
        """Test basic LocalArray creation."""
        self.assertEqual(self.larr_1d.shape, (7,))
        self.assertEqual(self.larr_1d.dist, ('b',))
        self.assertEqual(self.larr_1d.grid_shape, (4,))
        self.assertEqual(self.larr_1d.base_comm, self.comm)
        self.assertEqual(self.larr_1d.comm_size, 4)
        self.assertTrue(self.larr_1d.comm_rank in range(4))
        self.assertEqual(self.larr_1d.ndistdim, 1)
        self.assertEqual(self.larr_1d.distdims, (0,))
        self.assertEqual(self.larr_1d.comm.Get_topo(),
                         (list(self.larr_1d.grid_shape),
                          [0], [self.larr_1d.comm_rank]))
        self.assertEqual(len(self.larr_1d.maps), 1)
        self.assertEqual(self.larr_1d.shape, (7,))
        if self.larr_1d.comm_rank == 3:
            self.assertEqual(self.larr_1d.local_shape, (1,))
        else:
            self.assertEqual(self.larr_1d.local_shape, (2,))
        self.assertEqual(self.larr_1d.local_array.shape, self.larr_1d.local_shape)
        self.assertEqual(self.larr_1d.local_array.size, self.larr_1d.local_size)
        self.assertEqual(self.larr_1d.local_size, self.larr_1d.local_shape[0])
        self.assertEqual(self.larr_1d.local_array.dtype, self.larr_1d.dtype)

    @comm_null_passes
    def test_basic_2d(self):
        """Test basic LocalArray creation."""
        self.assertEqual(self.larr_2d.shape, (16,16))
        self.assertEqual(self.larr_2d.dist, ('b', 'n'))
        self.assertEqual(self.larr_2d.grid_shape, (4,))
        self.assertEqual(self.larr_2d.base_comm, self.comm)
        self.assertEqual(self.larr_2d.comm_size, 4)
        self.assertTrue(self.larr_2d.comm_rank in range(4))
        self.assertEqual(self.larr_2d.ndistdim, 1)
        self.assertEqual(self.larr_2d.distdims, (0,))
        self.assertEqual(self.larr_2d.comm.Get_topo(),
                         (list(self.larr_2d.grid_shape),
                          [0], [self.larr_2d.comm_rank]))
        self.assertEqual(len(self.larr_2d.maps), 2)
        self.assertEqual(self.larr_2d.grid_shape, (4,))
        self.assertEqual(self.larr_2d.shape, (16, 16))
        self.assertEqual(self.larr_2d.local_shape, (4, 16))
        self.assertEqual(self.larr_2d.local_size,
                         np.array(self.larr_2d.local_shape).prod())
        self.assertEqual(self.larr_2d.local_array.shape, self.larr_2d.local_shape)
        self.assertEqual(self.larr_2d.local_array.size, self.larr_2d.local_size)
        self.assertEqual(self.larr_2d.local_array.dtype, self.larr_2d.dtype)

    @comm_null_passes
    def test_localarray(self):
        """Can the local_array be set and get?"""
        self.larr_2d.get_localarray()
        la = np.random.random(self.larr_2d.local_shape)
        la = np.asarray(la, dtype=self.larr_2d.dtype)
        self.larr_2d.set_localarray(la)
        self.larr_2d.get_localarray()


class TestFromDimData(MpiTestCase):

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
        self.assertEqual(l0.local_shape, l1.local_shape)
        self.assertEqual(l0.local_array.shape, l1.local_array.shape)
        self.assertEqual(l0.local_array.dtype, l1.local_array.dtype)
        self.assertEqual(l0.local_shape, l1.local_shape)
        self.assertEqual(l0.local_size, l1.local_size)
        self.assertEqual(list(l0.maps[0].global_index),
                         list(l1.maps[0].global_index))
        self.assertEqual(list(l0.maps[0].local_index),
                         list(l1.maps[0].local_index))

    @comm_null_passes
    def test_block(self):
        dim0 = {
            "dist_type": 'b',
            "size": 16,
            "proc_grid_size": 4,
            }

        dim1 = {
            "dist_type": 'n',
            "size": 16,
            "proc_grid_size": None,
        }

        dim_data = (dim0, dim1)

        larr = da.LocalArray.from_dim_data(dim_data, comm=self.comm)
        expected = da.LocalArray((16,16), dist={0: 'b'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic(self):
        dim0 = {
            "dist_type": 'n',
            "size": 16,
            "proc_grid_size": None,
            }

        dim1 = {
            "dist_type": 'c',
            "size": 16,
            "proc_grid_size": 4,
            }

        dim_data = (dim0, dim1)

        larr = da.LocalArray.from_dim_data(dim_data, comm=self.comm)
        expected = da.LocalArray((16,16), dist={1: 'c'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic_and_block(self):
        dim0 = {
            "dist_type": 'c',
            "size": 16,
            "proc_grid_size": 2,
            }

        dim1 = {
            "dist_type": 'b',
            "size": 16,
            "proc_grid_size": 2,
            }

        dim_data = (dim0, dim1)

        larr = da.LocalArray.from_dim_data(dim_data, comm=self.comm)
        expected = da.LocalArray((16,16), dist={0: 'c', 1: 'b'},
                                 grid_shape=(2, 2), comm=self.comm)

        self.assert_alike(larr, expected)

    @unittest.skip('Not implemented.')
    def test_block_cyclic(self):
        dim0 = {"dist_type": 'c',
                "size": 16,
                "proc_grid_size": 4,
                "block_size": 2}

        dim1 = {"dist_type": 'n',
                "size": 16,
                "proc_grid_size": None}

        dim_data = (dim0, dim1)

        larr = da.LocalArray.from_dim_data(dim_data, comm=self.comm)
        expected = da.LocalArray((16,16), dist={0: 'c'}, block_size=2,
                                 grid_shape=(2, 2), comm=self.comm)

        self.assert_alike(larr, expected)


class TestGridShape(MpiTestCase):

    def get_comm_size(self):
        return 12

    @comm_null_passes
    def test_grid_shape(self):
        """Test various ways of setting the grid_shape."""
        self.larr = da.LocalArray((20,20), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (3,4))
        self.larr = da.LocalArray((2*10,6*10), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,6))
        self.larr = da.LocalArray((6*10,2*10), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (6,2))
        self.larr = da.LocalArray((100,10,300), dist=('b', 'n', 'c'), comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,6))
        self.larr = da.LocalArray((100,50,300), dist='b', comm=self.comm)
        self.assertEqual(self.larr.grid_shape, (2,2,3))


class TestDistMatrix(MpiTestCase):

    """Test the dist_matrix."""

    def get_comm_size(self):
        return 12

    @unittest.skip("Plot test.")
    @comm_null_passes
    def test_plot_dist_matrix(self):
        """Can we create and possibly plot a dist_matrix?"""
        la = da.LocalArray((10,10), dist=('c','c'), comm=self.comm)
        if self.comm.Get_rank() == 0:
            import pylab
            pylab.ion()
            pylab.matshow(la)
            pylab.colorbar()
            pylab.draw()
            pylab.show()


class TestLocalInd(MpiTestCase):

    """Test the computation of local indices."""

    @comm_null_passes
    def test_block_simple(self):
        """Can we compute local indices for a block distribution?"""
        la = da.LocalArray((4, 4), comm=self.comm)
        self.assertEqual(la.shape, (4, 4))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.local_shape, (1, 4))
        row_result = [(0, 0), (0, 1), (0, 2), (0, 3)]

        row = la.comm_rank
        calc_row_result = [la.global_to_local(row, col) for col in
                           range(la.shape[1])]
        self.assertEqual(row_result, calc_row_result)

    @comm_null_passes
    def test_block_complex(self):
        """Can we compute local indices for a block distribution?"""
        la = da.LocalArray((8, 2), comm=self.comm)
        self.assertEqual(la.shape, (8, 2))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.local_shape, (2, 2))
        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (1, 0), (1, 1)]
        elif la.comm_rank == 1:
            gis = [(2, 0), (2, 1), (3, 0), (3, 1)]
        elif la.comm_rank == 2:
            gis = [(4, 0), (4, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 3:
            gis = [(6, 0), (6, 1), (7, 0), (7, 1)]

        result = [la.global_to_local(*gi) for gi in gis]
        self.assertEqual(result, expected_lis)

    @comm_null_passes
    def test_cyclic_simple(self):
        """Can we compute local indices for a cyclic distribution?"""
        la = da.LocalArray((10,), dist={0: 'c'}, comm=self.comm)
        self.assertEqual(la.shape, (10,))
        self.assertEqual(la.grid_shape, (4,))

        if la.comm_rank == 0:
            gis = (0, 4, 8)
            self.assertEqual(la.local_shape, (3,))
            calc_result = [la.global_to_local(gi) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 1:
            gis = (1, 5, 9)
            self.assertEqual(la.local_shape, (3,))
            calc_result = [la.global_to_local(gi) for gi in gis]
            result = [(0,), (1,), (2,)]
        elif la.comm_rank == 2:
            gis = (2, 6)
            self.assertEqual(la.local_shape, (2,))
            calc_result = [la.global_to_local(gi) for gi in gis]
            result = [(0,), (1,)]
        elif la.comm_rank == 3:
            gis = (3, 7)
            self.assertEqual(la.local_shape, (2,))
            calc_result = [la.global_to_local(gi) for gi in gis]
            result = [(0,), (1,)]

        self.assertEqual(result, calc_result)

    @comm_null_passes
    def test_cyclic_complex(self):
        """Can we compute local indices for a cyclic distribution?"""
        la = da.LocalArray((8, 2), dist={0: 'c'}, comm=self.comm)
        self.assertEqual(la.shape, (8, 2))
        self.assertEqual(la.grid_shape, (4,))
        self.assertEqual(la.local_shape, (2, 2))

        expected_lis = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if la.comm_rank == 0:
            gis = [(0, 0), (0, 1), (4, 0), (4, 1)]
        elif la.comm_rank == 1:
            gis = [(1, 0), (1, 1), (5, 0), (5, 1)]
        elif la.comm_rank == 2:
            gis = [(2, 0), (2, 1), (6, 0), (6, 1)]
        elif la.comm_rank == 3:
            gis = [(3, 0), (3, 1), (7, 0), (7, 1)]

        result = [la.global_to_local(*gi) for gi in gis]
        self.assertEqual(result, expected_lis)


class TestGlobalInd(MpiTestCase):

    """Test the computation of global indices."""

    def round_trip(self, la):
        for indices in utils.multi_for([range(s) for s in la.local_shape]):
            gi = la.local_to_global(*indices)
            li = la.global_to_local(*gi)
            self.assertEqual(li,indices)

    @comm_null_passes
    def test_block(self):
        """Can we go from global to local indices and back for block?"""
        la = da.LocalArray((4,4), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_cyclic(self):
        """Can we go from global to local indices and back for cyclic?"""
        la = da.LocalArray((8,8), dist=('c', 'n'), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_crazy(self):
        """Can we go from global to local indices and back for a complex case?"""
        la = da.LocalArray((10,100,20), dist=('b', 'c', 'n'), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_global_limits_block(self):
        """Find the boundaries of a block distribution"""
        a = da.LocalArray((16, 16), dist=('b', 'n'), comm=self.comm)

        answers = [(0, 3), (4, 7), (8, 11), (12, 15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])

        answers = 4 * [(0, 15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])

    @comm_null_passes
    def test_global_limits_cyclic(self):
        """Find the boundaries of a cyclic distribution"""
        a = da.LocalArray((16,16), dist=('c', 'n'), comm=self.comm)
        answers = [(0,12),(1,13),(2,14),(3,15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])
        answers = 4*[(0,15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])


class TestIndexing(MpiTestCase):

    @comm_null_passes
    def test_indexing_0(self):
        """Can we get and set local elements for a simple dist?"""
        a = da.LocalArray((16,16), dist=('b', 'n'), comm=self.comm)
        b = da.LocalArray((16,16), dist=('b', 'n'), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_indexing_1(self):
        """Can we get and set local elements for a complex dist?"""
        a = da.LocalArray((16,16,2), dist=('c', 'b', 'n'), comm=self.comm)
        b = da.LocalArray((16,16,2), dist=('c', 'b', 'n'), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_pack_unpack_index(self):
        a = da.LocalArray((16,16,2), dist=('c', 'b', 'n'), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            packed_ind = a.pack_index(global_inds)
            self.assertEqual(global_inds, a.unpack_index(packed_ind))


class TestLocalArrayMethods(MpiTestCase):

    @comm_null_passes
    def test_asdist_like(self):
        """Test asdist_like for success and failure."""
        a = da.LocalArray((16,16), dist=('b', 'n'), comm=self.comm)
        b = da.LocalArray((16,16), dist=('b', 'n'), comm=self.comm)
        new_a = a.asdist_like(b)
        self.assertEqual(id(a),id(new_a))
        a = da.LocalArray((16,16), dist=('b', 'n'), comm=self.comm)
        b = da.LocalArray((16,16), dist=('n', 'b'), comm=self.comm)
        self.assertRaises(IncompatibleArrayError, a.asdist_like, b)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass

