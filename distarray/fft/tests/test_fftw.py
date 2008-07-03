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
from distarray import utils
from distarray.fft import py_fftw as fftw

class Test2dForward(unittest.TestCase):
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
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16),comm=comm,dtype='float64')
            except NullCommError:
                pass
            else:
                self.assertEquals(da.dist, ('b',None))
                result = fftw.fft2(da)
                numpy_result = np.fft.fft2(np.fromfunction(f,(16,16)))
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass