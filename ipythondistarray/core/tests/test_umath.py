import numpy as np

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
from ipythondistarray.core.nulldistarray import (
    allnull, 
    anynull, 
    nonenull,
    NullDistArray,
    null_like)


class TestUnaryUFunc(unittest.TestCase):
    
    def test_basic(self):
        """
        See if unary ufunc works for a DistArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = distarray.DistArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b = distarray.negative(a)
                self.assert_(np.all(a.local_array==-b.local_array))
                b = distarray.empty_like(a)
                b = distarray.negative(a, b)
                self.assert_(np.all(a.local_array==-b.local_array))
                a = distarray.DistArray((16,16), dtype='int32', comm=comm)
                b = distarray.DistArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, distarray.negative, b, a)


class TestBinaryUFunc(unittest.TestCase):
    
    def test_basic(self):
        """
        See if binary ufunc works for a DistArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                a = distarray.DistArray((16,16), dtype='int32', comm=comm)
                b = distarray.DistArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b.fill(1)
                c = distarray.add(a, b)
                self.assert_(np.all(c.local_array==2))
                c = distarray.empty_like(a)
                c = distarray.add(a, b, c)
                self.assert_(np.all(c.local_array==2))
                a = distarray.DistArray((16,16), dtype='int32', comm=comm)
                b = distarray.DistArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, distarray.add, a, b)
                a = distarray.DistArray((16,16), dtype='int32', comm=comm)
                b = distarray.DistArray((16,16), dtype='int32', comm=comm)
                c = distarray.DistArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, distarray.add, a, b, c)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass