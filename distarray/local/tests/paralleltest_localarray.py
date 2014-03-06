import unittest
import numpy as np

import distarray.local.denselocalarray as da
from distarray import utils
from distarray.testing import comm_null_passes, MpiTestCase
from distarray.local.error import InvalidDimensionError, IncompatibleArrayError


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

    @comm_null_passes
    def test_bad_distribution(self):
        """ Test that invalid distribution type fails as expected. """
        with self.assertRaises(TypeError):
            # Invalid distribution type 'x'.
            da.LocalArray((7,), dist={0: 'x'}, grid_shape=(4,),
                          comm=self.comm, buf=None)

    @comm_null_passes
    def test_no_grid_shape(self):
        """ Create array init when passing no grid_shape. """
        larr_nogrid = da.LocalArray((7,),
                                    comm=self.comm,
                                    buf=None)
        grid_shape = larr_nogrid.grid_shape
        # For 1D array as created here, we expect grid_shape
        # to just be the number of engines as a tuple.
        max_size = self.comm.Get_size()
        expected_grid_shape = (max_size,)
        self.assertEqual(grid_shape, expected_grid_shape)

    @comm_null_passes
    def test_bad_localarray(self):
        """ Test that setting a bad local array fails as expected. """
        self.larr_1d.get_localarray()
        local_shape = self.larr_1d.local_shape
        # Double dimension sizes to make an invalid shape.
        bad_shape = tuple(2 * size for size in local_shape)
        la = np.random.random(bad_shape)
        la = np.asarray(la, dtype=self.larr_1d.dtype)
        with self.assertRaises(ValueError):
            self.larr_1d.set_localarray(la)


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

    @comm_null_passes
    def test_bad_global_limits(self):
        """ Test that invalid global_limits fails as expected. """
        a = da.LocalArray((4, 4), comm=self.comm)
        with self.assertRaises(InvalidDimensionError):
            a.global_limits(-1)


class TestRankCoords(MpiTestCase):
    """ Test the rank <--> coords methods. """

    def round_trip(self, la, rank):
        """ Test that given a rank, we can get the coords,
        and then get back to the same rank. """
        coords = la.rank_to_coords(rank)
        # I am not sure what to expect for specific values for coords.
        # Therefore the specific return value is not checked.
        rank2 = la.coords_to_rank(coords)
        self.assertEqual(rank, rank2)

    @comm_null_passes
    def test_rank_coords(self):
        """ Test that we can go from rank to coords and back. """
        la = da.LocalArray((4,4), comm=self.comm)
        max_size = self.comm.Get_size()
        for rank in range(max_size):
            self.round_trip(la, rank=rank)


class TestArrayConversion(MpiTestCase):
    """ Test array conversion methods. """

    @comm_null_passes
    def more_setUp(self):
        self.int_larr = da.LocalArray((4,), dtype=int, comm=self.comm)
        self.int_larr.fill(3)

    @comm_null_passes
    def test_astype(self):
        """ Test that astype() works as expected. """
        # Convert int array to float.
        float_larr = self.int_larr.astype(float)
        for global_inds, value in da.ndenumerate(float_larr):
            self.assertEqual(value, 3.0)
            self.assertTrue(isinstance(value, float))
        # No type specification for a copy.
        int_larr2 = self.int_larr.astype(None)
        for global_inds, value in da.ndenumerate(int_larr2):
            self.assertEqual(value, 3)
            self.assertTrue(isinstance(value, int))

    @comm_null_passes
    def test_local_view(self):
        """ Test that local_views can be created as expected. """
        # Use dtype=None for same type.
        self.int_larr.local_view(None)
        # Use explicit dtype to change type.
        self.int_larr.local_view(float)
        # I am not sure what to expect for the values in the view,
        # so those are not checked here, so this is mainly a coverage test.

    @comm_null_passes
    def test_view(self):
        """ Test that views can be created as expected. """
        # Note this is mainly a coverage test for the same reason as above.
        # Use dtype=None for same type.
        self.int_larr.view(None)
        # Use explicit dtype to change type.
        self.int_larr.view(float)


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


class TestNotImplementedArrayMethods(MpiTestCase):
    """ Test that the not implemented functions can be called,
    and raise an exception. As these methods get implemented,
    they will start failing this test, and can be removed from it.
    Eventually this test case should become empty! """

    @comm_null_passes
    def more_setUp(self):
        # These would need real values for a real test,
        # but since we just check for not implemented,
        # this is not necessary here.
        self.larr1 = da.LocalArray((4,), comm=self.comm)
        self.larr2 = da.LocalArray((4, 4), comm=self.comm)
        self.larrb = da.LocalArray((4, 4), dtype=bool, comm=self.comm)
        self.larrc = da.LocalArray((4, 4), dtype=complex, comm=self.comm)

    @comm_null_passes
    def test_array_shape_manipulation(self):
        """ Array shape manipulation functions. """
        with self.assertRaises(NotImplementedError):
            self.larr2.reshape((8, 2))
        with self.assertRaises(NotImplementedError):
            self.larr2.redist((8, 2), newdist={0: 'n'})
        with self.assertRaises(NotImplementedError):
            self.larr2.resize((8, 4))
        with self.assertRaises(NotImplementedError):
            self.larr2.transpose(None)
        with self.assertRaises(NotImplementedError):
            self.larr2.swapaxes(0, 1)
        with self.assertRaises(NotImplementedError):
            self.larr2.flatten()
        with self.assertRaises(NotImplementedError):
            self.larr2.ravel()
        with self.assertRaises(NotImplementedError):
            self.larr2.squeeze()

    @comm_null_passes
    def test_array_item_selection_323(self):
        """ Array item selection functions 3.2.3. """
        with self.assertRaises(NotImplementedError):
            self.larr2.take([[0, 0], [1, 1]])
        with self.assertRaises(NotImplementedError):
            self.larr2.put([0, 3], [42, 27])
        with self.assertRaises(NotImplementedError):
            self.larr1.putmask([True, True, False, False], [12, 47])
        with self.assertRaises(NotImplementedError):
            self.larr1.repeat(3)
        with self.assertRaises(NotImplementedError):
            self.larr1.choose([[1], [2]])
        with self.assertRaises(NotImplementedError):
            self.larr1.sort()
        with self.assertRaises(NotImplementedError):
            self.larr1.argsort()
        with self.assertRaises(NotImplementedError):
            self.larr1.searchsorted([2, 3, 4])
        with self.assertRaises(NotImplementedError):
            self.larr1.nonzero()
        with self.assertRaises(NotImplementedError):
            self.larr2.compress([False, True, False, True])
        with self.assertRaises(NotImplementedError):
            self.larr2.diagonal()

    @comm_null_passes
    def test_array_item_selection_324(self):
        """ Array item selection functions 3.2.4. """
        with self.assertRaises(NotImplementedError):
            self.larr1.max()
        with self.assertRaises(NotImplementedError):
            self.larr1.argmax()
        with self.assertRaises(NotImplementedError):
            self.larr1.min()
        with self.assertRaises(NotImplementedError):
            self.larr1.argmin()
        with self.assertRaises(NotImplementedError):
            self.larr1.ptp()
        with self.assertRaises(NotImplementedError):
            self.larr1.clip(2, 5)
        with self.assertRaises(NotImplementedError):
            self.larrc.conj()
        with self.assertRaises(NotImplementedError):
            self.larr2.round()
        with self.assertRaises(NotImplementedError):
            self.larr2.trace()
        with self.assertRaises(NotImplementedError):
            self.larr2.cumsum()
        with self.assertRaises(NotImplementedError):
            self.larr2.prod()
        with self.assertRaises(NotImplementedError):
            self.larr2.cumprod()
        with self.assertRaises(NotImplementedError):
            self.larrb.all()
        with self.assertRaises(NotImplementedError):
            self.larrb.any()


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass

