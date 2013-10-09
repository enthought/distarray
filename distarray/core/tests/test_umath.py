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
from distarray.core import maps, denselocalarray


class TestUnaryUFunc(unittest.TestCase):

    def test_basic(self):
        """
        See if unary ufunc works for a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b = denselocalarray.negative(a)
                self.assertTrue(np.all(a.local_array==-b.local_array))
                b = denselocalarray.empty_like(a)
                b = denselocalarray.negative(a, b)
                self.assertTrue(np.all(a.local_array==-b.local_array))
                a = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = denselocalarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, denselocalarray.negative, b, a)


class TestBinaryUFunc(unittest.TestCase):

    def test_basic(self):
        """
        See if binary ufunc works for a LocalArray.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
            except NullCommError:
                pass
            else:
                a.fill(1)
                b.fill(1)
                c = denselocalarray.add(a, b)
                self.assertTrue(np.all(c.local_array==2))
                c = denselocalarray.empty_like(a)
                c = denselocalarray.add(a, b, c)
                self.assertTrue(np.all(c.local_array==2))
                a = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = denselocalarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, denselocalarray.add, a, b)
                a = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
                b = denselocalarray.LocalArray((16,16), dtype='int32', comm=comm)
                c = denselocalarray.LocalArray((20,20), dtype='int32', comm=comm)
                self.assertRaises(IncompatibleArrayError, denselocalarray.add, a, b, c)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
