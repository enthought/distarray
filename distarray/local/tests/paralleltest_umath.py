import unittest
import numpy as np

from distarray.local import denselocalarray
from distarray.local.error import IncompatibleArrayError
from distarray.testing import MpiTestCase, comm_null_passes


class TestUnaryUFunc(MpiTestCase):

    @comm_null_passes
    def test_negative(self):
        """See if unary ufunc works for a LocalArray."""
        a = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        a.fill(1)
        b = denselocalarray.negative(a)
        self.assertTrue(np.all(a.local_array==-b.local_array))

        b = denselocalarray.empty_like(a)
        b = denselocalarray.negative(a, b)
        self.assertTrue(np.all(a.local_array==-b.local_array))

        a = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        b = denselocalarray.LocalArray((20,20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError, denselocalarray.negative, b, a)


class TestBinaryUFunc(MpiTestCase):

    @comm_null_passes
    def test_add(self):
        """See if binary ufunc works for a LocalArray."""
        a = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        b = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        a.fill(1)
        b.fill(1)
        c = denselocalarray.add(a, b)
        self.assertTrue(np.all(c.local_array==2))

        c = denselocalarray.empty_like(a)
        c = denselocalarray.add(a, b, c)
        self.assertTrue(np.all(c.local_array==2))

        a = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        b = denselocalarray.LocalArray((20,20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError, denselocalarray.add, a, b)

        a = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        b = denselocalarray.LocalArray((16,16), dtype='int32', comm=self.comm)
        c = denselocalarray.LocalArray((20,20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError, denselocalarray.add, a, b, c)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
