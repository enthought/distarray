import unittest
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from ipythondistarray.core.error import *
from ipythondistarray.mpi.error import *
from ipythondistarray.mpi import mpibase
from ipythondistarray.mpi.mpibase import (
    MPI, 
    create_comm_of_size,
    create_comm_with_list)
from ipythondistarray.core import maps, distarray
from ipythondistarray import utils

class TestInit(unittest.TestCase):
    """
    Is the __init__ method working properly?
    """
    
    def test_basic(self):
        """
        Test basic DistArray creation.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                da = distarray.DistArray((16,16), grid_shape=(4,),comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(da.shape, (16,16))
                self.assertEquals(da.dist, ('b',None))
                self.assertEquals(da.grid_shape, (4,))
                self.assertEquals(da.base_comm, comm)
                self.assertEquals(da.comm_size, 4)
                self.assert_(da.comm_rank in range(4))
                self.assertEquals(da.ndistdim, 1)
                self.assertEquals(da.distdims, (0,))
                self.assertEquals(da.map_classes, (maps.BlockMap,))
                self.assertEquals(da.comm.Get_topo(), (da.grid_shape,(0,),(da.comm_rank,)))
                self.assertEquals(len(da.maps), 1)
                self.assertEquals(da.maps[0].local_shape, 4)
                self.assertEquals(da.maps[0].shape, 16)
                self.assertEquals(da.maps[0].grid_shape, 4)
                self.assertEquals(da.local_shape, (4,16))
                self.assertEquals(da.local_array.shape, da.local_shape)
                self.assertEquals(da.local_array.dtype, da.dtype)
                comm.Free()
    
    
    def test_localarray(self):
        """
        Can the local_array be set and get?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                da = distarray.DistArray((16,16), grid_shape=(4,), comm=comm)
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
            pass
        else:
            try:
                da = distarray.DistArray((20,20), dist='b', comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(da.grid_shape, (3,4))
                da = distarray.DistArray((2*10,6*10), dist='b', comm=comm)
                self.assertEquals(da.grid_shape, (2,6))
                da = distarray.DistArray((6*10,2*10), dist='b', comm=comm)
                self.assertEquals(da.grid_shape, (6,2))
                da = distarray.DistArray((100,10,300), dist=('b',None,'c'), comm=comm)
                self.assertEquals(da.grid_shape, (2,6))
                da = distarray.DistArray((100,50,300), dist='b', comm=comm)
                self.assertEquals(da.grid_shape, (2,2,3))                  
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
            pass
        else:
            try:
                da = distarray.DistArray((10,10), dist=('c','c'), comm=comm)
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
            pass
        else:
            try:
                da = distarray.DistArray((4,4),comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(da.shape,(4,4))
                self.assertEquals(da.grid_shape,(4,))
                row_result = [(0,0),(0,1),(0,2),(0,3)]
                for row in range(da.shape[0]):
                    calc_row_result = [da.local_ind(row,col) for col in range(da.shape[1])]
                    self.assertEquals(row_result, calc_row_result)
                comm.Free()
    
    
    def test_cyclic(self):
        """
        Can we compute local incides for a CyclicMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                da = distarray.DistArray((8,8),dist={0:'c'},comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(da.shape,(8,8))
                self.assertEquals(da.grid_shape,(4,))
                self.assertEquals(da.map_classes, (maps.CyclicMap,))
                result = utils.outer_zip(4*(0,)+4*(1,),range(8))
                calc_result = [[da.local_ind(row,col) for col in range(da.shape[1])] for row in range(da.shape[0])]
                self.assertEquals(result,calc_result)
                comm.Free()


class TestGlobalInd(unittest.TestCase):
    """
    Test the computation of global indices.
    """
    
    def round_trip(self, da):
        for indices in utils.multi_for( [xrange(s) for s in da.shape] ):
            li = da.local_ind(*indices)
            owner_rank = da.owner_rank(*indices)
            gi = da.global_ind(owner_rank,*li)
            self.assertEquals(gi,indices)
    
    
    def test_block(self):
        """
        Can we go from global to local indices and back for BlockMap?
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                da = distarray.DistArray((4,4),comm=comm)
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
            pass
        else:
            try:
                da = distarray.DistArray((8,8),dist=('c',None),comm=comm)
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
            pass
        else:
            try:
                da = distarray.DistArray((10,100,20),dist=('b','c',None),comm=comm)
            except NullCommError:
                pass
            else:
                self.round_trip(da)
                comm.Free()


class TestDistArrayMethods(unittest.TestCase):
    
    def test_asdist_like(self):
        """
        Test asdist_like for success and failure.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = distarray.DistArray((16,16), dist=('b',None),comm=comm)
                b = distarray.DistArray((16,16), dist=('b',None),comm=comm)
            except NullCommError:
                pass
            else:
                new_a = a.asdist_like(b)
                self.assertEquals(id(a),id(new_a))
                a = distarray.DistArray((16,16), dist=('b',None),comm=comm)
                b = distarray.DistArray((16,16), dist=(None,'b'),comm=comm)
                self.assertRaises(IncompatibleArrayError, a.asdist_like, b)
                comm.Free()


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
	

