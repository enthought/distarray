import unittest
import numpy as np

from distarray.core.error import (DistError, IncompatibleArrayError,
                                  NullCommError)
from distarray.mpi.error import InvalidCommSizeError
from distarray.mpi import mpibase
from distarray.mpi.mpibase import (MPI, create_comm_of_size,
                                   create_comm_with_list)
from distarray.core import maps, denselocalarray
from distarray import utils


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
                        pylab.matshow(a)
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


class TestDistArrayMethods(unittest.TestCase):

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


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
