import unittest
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from distarray.core.error import *
from distarray.mpi.error import *
from distarray.mpi import mpibase
from distarray.mpi.mpibase import (
    MPI, 
    create_comm_of_size,
    create_comm_with_list)
from distarray.core import maps_fast as maps, densedistarray


class TestFunctions(unittest.TestCase):
    
    def test_arecompatible(self):
        """
        Test if two DistArrays are compatible.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = densedistarray.DistArray((16,16), dtype='int64', comm=comm)
                b = densedistarray.DistArray((16,16), dtype='float32', comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(densedistarray.arecompatible(a,b), True)
                a = densedistarray.DistArray((16,16), dtype='int64', dist='c', comm=comm)
                b = densedistarray.DistArray((16,16), dtype='float32', dist='b', comm=comm)
                self.assertEquals(densedistarray.arecompatible(a,b), False)                
                comm.Free()
    
    def test_fromfunction(self):
        """
        Can we build an array using fromfunction and a trivial function.
        """
        def f(*global_inds):
            return 1.0
        
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = densedistarray.fromfunction(f, (16,16), dtype='int64', dist=('b','c'), comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(a.shape, (16,16))
                self.assertEquals(a.dtype, np.dtype('int64'))
                for global_inds, value in densedistarray.ndenumerate(a):
                    self.assertEquals(1.0, value)
                comm.Free()
    
    def test_fromfunction_complicated(self):
        """
        Can we build an array using fromfunction and a nontrivial function.
        """
        def f(*global_inds):
            return sum(global_inds)
        
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = densedistarray.fromfunction(f, (16,16), dtype='int64', dist=('b','c'), comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(a.shape, (16,16))
                self.assertEquals(a.dtype, np.dtype('int64'))
                for global_inds, value in densedistarray.ndenumerate(a):
                    self.assertEquals(sum(global_inds), value)
                comm.Free()




if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass