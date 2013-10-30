import unittest
import numpy as np
import distarray.core as dc

from numpy.testing import assert_array_equal
from distarray import utils
from distarray.mpi.mpibase import create_comm_of_size
from distarray.core import maps, denselocalarray
from distarray.core.error import (DistError, IncompatibleArrayError,
                                  NullCommError)
from distarray.mpi.error import InvalidCommSizeError


class TestInit(unittest.TestCase):
    """
    Is the __init__ method working properly?
    """

    def test_basic(self):
        """
        Test basic LocalArray creation.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((16,16), grid_shape=(4,),comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEqual(da.shape, (16,16))
                self.assertEqual(da.dist, ('b',None))
                self.assertEqual(da.grid_shape, (4,))
                self.assertEqual(da.base_comm, comm)
                self.assertEqual(da.comm_size, 4)
                self.assertTrue(da.comm_rank in range(4))
                self.assertEqual(da.ndistdim, 1)
                self.assertEqual(da.distdims, (0,))
                self.assertEqual(da.map_classes, (maps.BlockMap,))
                self.assertEqual(da.comm.Get_topo(), (list(da.grid_shape),[0],[da.comm_rank]))
                self.assertEqual(len(da.maps), 1)
                self.assertEqual(da.maps[0].local_shape, 4)
                self.assertEqual(da.maps[0].shape, 16)
                self.assertEqual(da.maps[0].grid_shape, 4)
                self.assertEqual(da.local_shape, (4,16))
                self.assertEqual(da.local_array.shape, da.local_shape)
                self.assertEqual(da.local_array.dtype, da.dtype)
                comm.Free()


    def test_localarray(self):
        """
        Can the local_array be set and get?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((16,16), grid_shape=(4,), comm=comm)
            except NullCommError:
                pass
            else:
                da.get_localarray()
                la = np.random.random(da.local_shape)
                la = np.asarray(la, dtype=da.dtype)
                da.set_localarray(la)
                new_la = da.get_localarray()
                comm.Free()


    def test_grid_shape(self):
        """
        Test various ways of setting the grid_shape.
        """
        try:
            comm = create_comm_of_size(12)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((20,20), dist='b', comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEqual(da.grid_shape, (3,4))
                da = denselocalarray.LocalArray((2*10,6*10), dist='b', comm=comm)
                self.assertEqual(da.grid_shape, (2,6))
                da = denselocalarray.LocalArray((6*10,2*10), dist='b', comm=comm)
                self.assertEqual(da.grid_shape, (6,2))
                da = denselocalarray.LocalArray((100,10,300), dist=('b',None,'c'), comm=comm)
                self.assertEqual(da.grid_shape, (2,6))
                da = denselocalarray.LocalArray((100,50,300), dist='b', comm=comm)
                self.assertEqual(da.grid_shape, (2,2,3))
                comm.Free()


class TestDistMatrix(unittest.TestCase):
    """
    Test the dist_matrix.
    """

    def test_plot_dist_matrix(self):
        """
        Can we create and possibly plot a dist_matrix?
        """
        try:
            comm = create_comm_of_size(12)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((10,10), dist=('c','c'), comm=comm)
            except NullCommError:
                pass
            else:
                if False:
                    if comm.Get_rank()==0:
                        import pylab
                        pylab.ion()
                        pylab.matshow(da)
                        pylab.colorbar()
                        pylab.draw()
                        pylab.show()
                comm.Free()


class TestLocalInd(unittest.TestCase):
    """
    Test the computation of local indices.
    """

    def test_block(self):
        """
        Can we compute local incides for a BlockMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((4,4),comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEqual(da.shape,(4,4))
                self.assertEqual(da.grid_shape,(4,))
                row_result = [(0,0),(0,1),(0,2),(0,3)]
                for row in range(da.shape[0]):
                    calc_row_result = [da.global_to_local(row,col) for col in range(da.shape[1])]
                    self.assertEqual(row_result, calc_row_result)
                comm.Free()

    def test_cyclic(self):
        """
        Can we compute local incides for a CyclicMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((8,8),dist={0:'c'},comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEqual(da.shape,(8,8))
                self.assertEqual(da.grid_shape,(4,))
                self.assertEqual(da.map_classes, (maps.CyclicMap,))
                result = utils.outer_zip(4*(0,)+4*(1,),list(range(8)))
                calc_result = [[da.global_to_local(row,col) for col in range(da.shape[1])] for row in range(da.shape[0])]
                self.assertEqual(result,calc_result)
                comm.Free()


class TestGlobalInd(unittest.TestCase):
    """
    Test the computation of global indices.
    """

    def round_trip(self, da):
        for indices in utils.multi_for( [range(s) for s in da.shape] ):
            li = da.global_to_local(*indices)
            owner_rank = da.owner_rank(*indices)
            gi = da.local_to_global(owner_rank,*li)
            self.assertEqual(gi,indices)

    def test_block(self):
        """
        Can we go from global to local indices and back for BlockMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((4,4),comm=comm)
            except NullCommError:
                pass
            else:
                self.round_trip(da)
                comm.Free()

    def test_cyclic(self):
        """
        Can we go from global to local indices and back for CyclicMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((8,8),dist=('c',None),comm=comm)
            except NullCommError:
                pass
            else:
                self.round_trip(da)
                comm.Free()

    def test_crazy(self):
        """
        Can we go from global to local indices and back for a complex case?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                da = denselocalarray.LocalArray((10,100,20),dist=('b','c',None),comm=comm)
            except NullCommError:
                pass
            else:
                self.round_trip(da)
                comm.Free()

    def test_global_limits(self):
        """Find the boundaries of a block distribution or no distribution"""
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
                b = denselocalarray.LocalArray((16,16), dist=('c',None),comm=comm)
            except NullCommError:
                pass
            else:
                answers = [(0,3),(4,7),(8,11),(12,15)]
                limits = a.global_limits(0)
                self.assertEqual(limits, answers[a.comm_rank])
                answers = 4*[(0,15)]
                limits = a.global_limits(1)
                self.assertEqual(limits, answers[a.comm_rank])
                self.assertRaises(DistError, b.global_limits, 0)
                comm.Free()


class TestIndexing(unittest.TestCase):

    def test_indexing0(self):
        """Can we get and set local elements for a simple dist?"""
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
                b = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
            except NullCommError:
                pass
            else:
                for global_inds, value in denselocalarray.ndenumerate(a):
                    a[global_inds] = 0.0
                for global_inds, value in denselocalarray.ndenumerate(a):
                    b[global_inds] = a[global_inds]
                for global_inds, value in denselocalarray.ndenumerate(a):
                    self.assertEqual(b[global_inds],a[global_inds])
                    self.assertEqual(a[global_inds],0.0)
                comm.Free()

    def test_indexing1(self):
        """Can we get and set local elements for a complex dist?"""
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16,2), dist=('c','b',None),comm=comm)
                b = denselocalarray.LocalArray((16,16,2), dist=('c','b',None),comm=comm)
            except NullCommError:
                pass
            else:
                for global_inds, value in denselocalarray.ndenumerate(a):
                    a[global_inds] = 0.0
                for global_inds, value in denselocalarray.ndenumerate(a):
                    b[global_inds] = a[global_inds]
                for global_inds, value in denselocalarray.ndenumerate(a):
                    self.assertEqual(b[global_inds],a[global_inds])
                    self.assertEqual(a[global_inds],0.0)
                comm.Free()

    def test_pack_unpack_index(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16,2), dist=('c','b',None),comm=comm)
            except NullCommError:
                pass
            else:
                for global_inds, value in denselocalarray.ndenumerate(a):
                    packed_ind = a.pack_index(global_inds)
                    self.assertEqual(global_inds, a.unpack_index(packed_ind))
                comm.Free()


class TestLocalArrayMethods(unittest.TestCase):

    def test_asdist_like(self):
        """
        Test asdist_like for success and failure.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
                b = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
            except NullCommError:
                pass
            else:
                new_a = a.asdist_like(b)
                self.assertEqual(id(a),id(new_a))
                a = denselocalarray.LocalArray((16,16), dist=('b',None),comm=comm)
                b = denselocalarray.LocalArray((16,16), dist=(None,'b'),comm=comm)
                self.assertRaises(IncompatibleArrayError, a.asdist_like, b)
                comm.Free()


def add_checkers(cls, ops):
    """Add a test method for all of the `ops`"""
    for op in ops:
        fn_name = "test_" + op.__name__
        fn_value = lambda self: self.check_op(op)
        setattr(cls, fn_name, fn_value)


class TestLocalArrayUnaryOperations(unittest.TestCase):

    def check_op(self, op):
        """Check unary operation for success.

        Check the one- and two-arg ufunc versions as well as the method
        version attached to a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                x = denselocalarray.ones((16,16), dist=('b',None), comm=comm,
                                         dtype='uint8')
                y = denselocalarray.ones((16,16), dist=('b',None), comm=comm,
                                         dtype='uint8')
            except NullCommError:
                pass
            else:
                result0 = op(x)  # standard form
                op(x, y=y)  # two-arg form
                result1 = eval("x." + op.__name__ + "()")  # method form
                assert_array_equal(result0.local_array, y.local_array)
                assert_array_equal(result0.local_array, result_1.local_array)
                comm.Free()

uops = (dc.absolute, dc.arccos, dc.arccosh, dc.arcsin, dc.arcsinh, dc.arctan,
        dc.arctanh, dc.conjugate, dc.cos, dc.cosh, dc.exp, dc.expm1, dc.invert,
        dc.log, dc.log10, dc.log1p, dc.negative, dc.reciprocal, dc.rint,
        dc.sign, dc.sin, dc.sinh, dc.sqrt, dc.square, dc.tan, dc.tanh)
add_checkers(TestLocalArrayUnaryOperations, uops)


class TestLocalArrayBinaryOperations(unittest.TestCase):

    def check_op(self, op):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                x1 = denselocalarray.ones((16,16), dist=('b',None), comm=comm,
                                          dtype='uint8')
                x2 = denselocalarray.ones((16,16), dist=('b',None), comm=comm,
                                          dtype='uint8')
                y = denselocalarray.ones((16,16), dist=('b',None), comm=comm,
                                         dtype='uint8')
            except NullCommError:
                pass
            else:
                result0 = op(x1, x2)  # standard form
                op(x1, x2, y=y) # three-arg form
                result1 = eval("x1." + op.__name__ + "(x2)")  # method form
                assert_array_equal(result0.local_array, y.local_array)
                assert_array_equal(result0.local_array, result1.local_array)
                comm.Free()


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
