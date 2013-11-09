import unittest
import numpy as np

import distarray.core as dc
import distarray.core.denselocalarray as da

from numpy.testing import assert_array_equal
from distarray.testing import comm_null_passes, MpiTestCase
from distarray import utils
from distarray.core import maps
from distarray.core.error import IncompatibleArrayError


class TestInit(MpiTestCase):

    """Is the __init__ method working properly?"""

    @comm_null_passes
    def more_setUp(self):
        self.larr = da.LocalArray((16,16), grid_shape=(4,), comm=self.comm,
                                  buf=None)

    @comm_null_passes
    def test_basic(self):
        """Test basic LocalArray creation."""
        self.assertEqual(self.larr.shape, (16,16))
        self.assertEqual(self.larr.dist, ('b',None))
        self.assertEqual(self.larr.grid_shape, (4,))
        self.assertEqual(self.larr.base_comm, self.comm)
        self.assertEqual(self.larr.comm_size, 4)
        self.assertTrue(self.larr.comm_rank in range(4))
        self.assertEqual(self.larr.ndistdim, 1)
        self.assertEqual(self.larr.distdims, (0,))
        self.assertEqual(self.larr.map_classes, (maps.BlockMap,))
        self.assertEqual(self.larr.comm.Get_topo(),
                (list(self.larr.grid_shape),[0],[self.larr.comm_rank]))
        self.assertEqual(len(self.larr.maps), 1)
        self.assertEqual(self.larr.maps[0].local_shape, 4)
        self.assertEqual(self.larr.maps[0].shape, 16)
        self.assertEqual(self.larr.maps[0].grid_shape, 4)
        self.assertEqual(self.larr.local_shape, (4,16))
        self.assertEqual(self.larr.local_array.shape, self.larr.local_shape)
        self.assertEqual(self.larr.local_array.dtype, self.larr.dtype)

    @comm_null_passes
    def test_localarray(self):
        """Can the local_array be set and get?"""
        self.larr.get_localarray()
        la = np.random.random(self.larr.local_shape)
        la = np.asarray(la, dtype=self.larr.dtype)
        self.larr.set_localarray(la)
        new_la = self.larr.get_localarray()


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
        self.assertEqual(l0.map_classes, l1.map_classes)
        self.assertEqual(l0.comm.Get_topo(), l1.comm.Get_topo())
        self.assertEqual(len(l0.maps), len(l1.maps))
        self.assertEqual(l0.maps[0].local_shape, l1.maps[0].local_shape)
        self.assertEqual(l0.maps[0].shape, l1.maps[0].shape)
        self.assertEqual(l0.maps[0].grid_shape, l1.maps[0].grid_shape)
        self.assertEqual(l0.local_shape, l1.local_shape)
        self.assertEqual(l0.local_array.shape, l1.local_array.shape)
        self.assertEqual(l0.local_array.dtype, l1.local_array.dtype)

    @comm_null_passes
    def test_block(self):
        dim0 = {"disttype": 'b',
                "datasize": 16,
                "gridsize": 4}

        dim1 = {"disttype": None,
                "datasize": 16,
                "gridsize": None}

        dimdata = (dim0, dim1)

        larr = da.LocalArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.LocalArray((16,16), dist={0: 'b'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic(self):
        dim0 = {"disttype": None,
                "datasize": 16,
                "gridsize": None}

        dim1 = {"disttype": 'c',
                "datasize": 16,
                "gridsize": 4}

        dimdata = (dim0, dim1)

        larr = da.LocalArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.LocalArray((16,16), dist={1: 'c'}, grid_shape=(4,),
                                 comm=self.comm)

        self.assert_alike(larr, expected)

    @comm_null_passes
    def test_cyclic_and_block(self):
        dim0 = {"disttype": 'c',
                "datasize": 16,
                "gridsize": 2}

        dim1 = {"disttype": 'b',
                "datasize": 16,
                "gridsize": 2}

        dimdata = (dim0, dim1)

        larr = da.LocalArray.from_dimdata(dimdata, comm=self.comm)
        expected = da.LocalArray((16,16), dist={0: 'c', 1: 'b'},
                                 grid_shape=(2, 2), comm=self.comm)

        self.assert_alike(larr, expected)

    @unittest.skip("Not yet implemented.")
    def test_block_cyclic(self):
        dim0 = {"disttype": 'bc',
                "datasize": 16,
                "gridsize": 4,
                "blocksize": 2}

        dim1 = {"disttype": None,
                "datasize": 16,
                "gridsize": None}

        dimdata = (dim0, dim1)

        da.LocalArray.from_dimdata(dimdata, comm=self.comm)


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
        self.larr = da.LocalArray((100,10,300), dist=('b',None,'c'), comm=self.comm)
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
    def test_block(self):
        """Can we compute local indices for a BlockMap?"""
        la = da.LocalArray((4,4), comm=self.comm)
        self.assertEqual(la.shape,(4,4))
        self.assertEqual(la.grid_shape,(4,))
        row_result = [(0,0),(0,1),(0,2),(0,3)]
        for row in range(la.shape[0]):
            calc_row_result = [la.global_to_local(row,col) for col in
                               range(la.shape[1])]
            self.assertEqual(row_result, calc_row_result)

    @comm_null_passes
    def test_cyclic(self):
        """Can we compute local indices for a CyclicMap?"""
        la = da.LocalArray((8,8),dist={0:'c'},comm=self.comm)
        self.assertEqual(la.shape,(8,8))
        self.assertEqual(la.grid_shape,(4,))
        self.assertEqual(la.map_classes, (maps.CyclicMap,))
        result = utils.outer_zip(4*(0,)+4*(1,),list(range(8)))
        calc_result = [[la.global_to_local(row,col) for col in
                        range(la.shape[1])] for row in range(la.shape[0])]
        self.assertEqual(result,calc_result)


class TestGlobalInd(MpiTestCase):

    """Test the computation of global indices."""

    def round_trip(self, la):
        for indices in utils.multi_for( [range(s) for s in la.shape] ):
            li = la.global_to_local(*indices)
            owner_rank = la.owner_rank(*indices)
            gi = la.local_to_global(owner_rank,*li)
            self.assertEqual(gi,indices)

    @comm_null_passes
    def test_block(self):
        """Can we go from global to local indices and back for BlockMap?"""
        la = da.LocalArray((4,4), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_cyclic(self):
        """Can we go from global to local indices and back for CyclicMap?"""
        la = da.LocalArray((8,8), dist=('c',None), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_crazy(self):
        """Can we go from global to local indices and back for a complex case?"""
        la = da.LocalArray((10,100,20), dist=('b','c',None), comm=self.comm)
        self.round_trip(la)

    @comm_null_passes
    def test_global_limits_block(self):
        """Find the boundaries of a block distribution"""
        a = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        answers = [(0,3),(4,7),(8,11),(12,15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])
        answers = 4*[(0,15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])

    @comm_null_passes
    def test_global_limits_cyclic(self):
        """Find the boundaries of a cyclic distribution"""
        a = da.LocalArray((16,16), dist=('c',None), comm=self.comm)
        answers = [(0,12),(1,13),(2,14),(3,15)]
        limits = a.global_limits(0)
        self.assertEqual(limits, answers[a.comm_rank])
        answers = 4*[(0,15)]
        limits = a.global_limits(1)
        self.assertEqual(limits, answers[a.comm_rank])


class TestIndexing(MpiTestCase):

    @comm_null_passes
    def test_indexing0(self):
        """Can we get and set local elements for a simple dist?"""
        a = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        b = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_indexing1(self):
        """Can we get and set local elements for a complex dist?"""
        a = da.LocalArray((16,16,2), dist=('c','b',None), comm=self.comm)
        b = da.LocalArray((16,16,2), dist=('c','b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            a[global_inds] = 0.0
        for global_inds, value in da.ndenumerate(a):
            b[global_inds] = a[global_inds]
        for global_inds, value in da.ndenumerate(a):
            self.assertEqual(b[global_inds],a[global_inds])
            self.assertEqual(a[global_inds],0.0)

    @comm_null_passes
    def test_pack_unpack_index(self):
        a = da.LocalArray((16,16,2), dist=('c','b',None), comm=self.comm)
        for global_inds, value in da.ndenumerate(a):
            packed_ind = a.pack_index(global_inds)
            self.assertEqual(global_inds, a.unpack_index(packed_ind))


class TestLocalArrayMethods(MpiTestCase):

    @comm_null_passes
    def test_asdist_like(self):
        """Test asdist_like for success and failure."""
        a = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        b = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        new_a = a.asdist_like(b)
        self.assertEqual(id(a),id(new_a))
        a = da.LocalArray((16,16), dist=('b',None), comm=self.comm)
        b = da.LocalArray((16,16), dist=(None,'b'), comm=self.comm)
        self.assertRaises(IncompatibleArrayError, a.asdist_like, b)


def add_checkers(cls, ops):
    """Add a test method to `cls` for all `ops`

    Parameters
    ----------
    cls : a Test class
    ops : an iterable of functions
        Functions to check with self.check_op(self, op)
    """
    for op in ops:
        fn_name = "test_" + op.__name__
        fn_value = lambda self: self.check_op(op)
        setattr(cls, fn_name, fn_value)


class TestLocalArrayUnaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check unary operation for success.

        Check the one- and two-arg ufunc versions as well as the method
        version attached to a LocalArray.
        """
        x = da.ones((16,16), dist=('b',None), comm=self.comm)
        y = da.ones((16,16), dist=('b',None), comm=self.comm)
        result0 = op(x)  # standard form
        op(x, y=y)  # two-arg form
        assert_array_equal(result0.local_array, y.local_array)

uops = (dc.absolute, dc.arccos, dc.arccosh, dc.arcsin, dc.arcsinh, dc.arctan,
        dc.arctanh, dc.conjugate, dc.cos, dc.cosh, dc.exp, dc.expm1, dc.invert,
        dc.log, dc.log10, dc.log1p, dc.negative, dc.reciprocal, dc.rint,
        dc.sign, dc.sin, dc.sinh, dc.sqrt, dc.square, dc.tan, dc.tanh)
add_checkers(TestLocalArrayUnaryOperations, uops)


class TestLocalArrayBinaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        x1 = da.ones((16,16), dist=('b',None), comm=self.comm)
        x2 = da.ones((16,16), dist=('b',None), comm=self.comm)
        y = da.ones((16,16), dist=('b',None), comm=self.comm)
        result0 = op(x1, x2)  # standard form
        op(x1, x2, y=y) # three-arg form
        assert_array_equal(result0.local_array, y.local_array)


bops = (dc.add, dc.arctan2, dc.bitwise_and, dc.bitwise_or, dc.bitwise_xor,
        dc.divide, dc.floor_divide, dc.fmod, dc.hypot, dc.left_shift, dc.mod,
        dc.multiply, dc.power, dc.remainder, dc.right_shift, dc.subtract,
        dc.true_divide)
add_checkers(TestLocalArrayBinaryOperations, bops)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
