import numpy as np

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
from distarray.core import maps, densedistarray


class TestUnaryUFunc(unittest.TestCase):
    
    def test_basic(self):
        """
        See if unary ufunc works for a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b = densedistarray.negative(a)
                self.assert_(np.all(a.local_array==-b.local_array))
                b = densedistarray.empty_like(a)
                b = densedistarray.negative(a, b)
                self.assert_(np.all(a.local_array==-b.local_array))
                a = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = densedistarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, densedistarray.negative, b, a)


class TestBinaryUFunc(unittest.TestCase):
    
    def test_basic(self):
        """
        See if binary ufunc works for a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b.fill(1)
                c = densedistarray.add(a, b)
                self.assert_(np.all(c.local_array==2))
                c = densedistarray.empty_like(a)
                c = densedistarray.add(a, b, c)
                self.assert_(np.all(c.local_array==2))
                a = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = densedistarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, densedistarray.add, a, b)
                a = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = densedistarray.LocalArray((16,16), dtype='int32', comm=comm)
                c = densedistarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, densedistarray.add, a, b, c)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
