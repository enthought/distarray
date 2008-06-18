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
from distarray.core import maps_fast as maps, distarray


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
                a = distarray.DistArray((16,16), dtype='int64', comm=comm)
                b = distarray.DistArray((16,16), dtype='float32', comm=comm)
            except NullCommError:
                pass
            else:
                self.assertEquals(distarray.arecompatible(a,b), True)
                a = distarray.DistArray((16,16), dtype='int64', dist='c', comm=comm)
                b = distarray.DistArray((16,16), dtype='float32', dist='b', comm=comm)
                self.assertEquals(distarray.arecompatible(a,b), False)                
                comm.Free()




if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass