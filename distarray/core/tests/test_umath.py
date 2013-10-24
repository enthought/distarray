import unittest
import numpy as np

from distarray.mpi.mpibase import create_comm_of_size
from distarray.core import denselocalarray
from distarray.mpi.error import InvalidCommSizeError
from distarray.core.error import IncompatibleArrayError, NullCommError


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
