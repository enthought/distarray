import unittest
import numpy as np
from numpy.testing import assert_array_equal

import distarray.local as dc
import distarray.local.denselocalarray as da
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


def add_checkers(cls, ops):
    """Add a test method to `cls` for all `ops`

    Parameters
    ----------
    cls : a Test class
    ops : an iterable of functions
        Functions to check with self.check_op(self, op)
    """
    for op in ops:
        fn_name = "test_" + op.__name__
        fn_value = lambda self: self.check_op(op)
        setattr(cls, fn_name, fn_value)


class TestLocalArrayUnaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check unary operation for success.

        Check the one- and two-arg ufunc versions as well as the method
        version attached to a LocalArray.
        """
        x = da.ones((16,16), dist=('b',None), comm=self.comm)
        y = da.ones((16,16), dist=('b',None), comm=self.comm)
        result0 = op(x)  # standard form
        op(x, y=y)  # two-arg form
        assert_array_equal(result0.local_array, y.local_array)

uops = (dc.absolute, dc.arccos, dc.arccosh, dc.arcsin, dc.arcsinh, dc.arctan,
        dc.arctanh, dc.conjugate, dc.cos, dc.cosh, dc.exp, dc.expm1, dc.invert,
        dc.log, dc.log10, dc.log1p, dc.negative, dc.reciprocal, dc.rint,
        dc.sign, dc.sin, dc.sinh, dc.sqrt, dc.square, dc.tan, dc.tanh)

add_checkers(TestLocalArrayUnaryOperations, uops)


class TestLocalArrayBinaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        x1 = da.ones((16,16), dist=('b',None), comm=self.comm)
        x2 = da.ones((16,16), dist=('b',None), comm=self.comm)
        y = da.ones((16,16), dist=('b',None), comm=self.comm)
        result0 = op(x1, x2)  # standard form
        op(x1, x2, y=y) # three-arg form
        assert_array_equal(result0.local_array, y.local_array)


bops = (dc.add, dc.arctan2, dc.bitwise_and, dc.bitwise_or, dc.bitwise_xor,
        dc.divide, dc.floor_divide, dc.fmod, dc.hypot, dc.left_shift, dc.mod,
        dc.multiply, dc.power, dc.remainder, dc.right_shift, dc.subtract,
        dc.true_divide)

add_checkers(TestLocalArrayBinaryOperations, bops)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
