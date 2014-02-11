import warnings
import unittest
import numpy as np
from numpy.testing import assert_array_equal

import distarray.remote as dc
import distarray.remote.denseremotearray as da
from distarray.remote import denseremotearray
from distarray.remote.error import IncompatibleArrayError
from distarray.testing import MpiTestCase, comm_null_passes


class TestUnaryUFunc(MpiTestCase):

    @comm_null_passes
    def test_negative(self):
        """See if unary ufunc works for a RemoteArray."""
        a = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        a.fill(1)
        b = denseremotearray.negative(a)
        self.assertTrue(np.all(a.remote_array == -b.remote_array))

        b = denseremotearray.empty_like(a)
        b = denseremotearray.negative(a, b)
        self.assertTrue(np.all(a.remote_array == -b.remote_array))

        a = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        b = denseremotearray.RemoteArray((20, 20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError,
                          denseremotearray.negative,
                          b, a)


class TestBinaryUFunc(MpiTestCase):

    @comm_null_passes
    def test_add(self):
        """See if binary ufunc works for a RemoteArray."""
        a = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        b = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        a.fill(1)
        b.fill(1)
        c = denseremotearray.add(a, b)
        self.assertTrue(np.all(c.remote_array == 2))

        c = denseremotearray.empty_like(a)
        c = denseremotearray.add(a, b, c)
        self.assertTrue(np.all(c.remote_array == 2))

        a = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        b = denseremotearray.RemoteArray((20, 20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError, denseremotearray.add, a, b)

        a = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        b = denseremotearray.RemoteArray((16, 16), dtype='int32', comm=self.comm)
        c = denseremotearray.RemoteArray((20, 20), dtype='int32', comm=self.comm)
        self.assertRaises(IncompatibleArrayError, denseremotearray.add, a, b, c)


def add_checkers(cls, ops, bad_ops):
    """Add a test method to `cls` for all `ops`

    Parameters
    ----------
    cls : a Test class
    ops : an iterable of functions
        Functions to check with self.check_op(self, op)
    bad_ops : an iterable of functions that are in ops which should be
        skipped.
    """
    msg = ("This operation does not work with the default "
           "testing data.")

    def check(op):
        return lambda self: self.check_op(op)

    for op in ops:
        fn_name = "test_" + op.__name__
        if op in bad_ops:
            setattr(cls, fn_name, lambda _: unittest.skip(msg))
        else:
            setattr(cls, fn_name, check(op))


class TestRemoteArrayUnaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check unary operation for success.

        Check the one- and two-arg ufunc versions as well as the method
        version attached to a RemoteArray.
        """
        x = da.ones((16, 16), dist=('b', None), comm=self.comm)
        y = da.ones((16, 16), dist=('b', None), comm=self.comm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result0 = op(x, casting='unsafe')  # standard form
            op(x, y=y, casting='unsafe')  # two-arg form
        assert_array_equal(result0.remote_array, y.remote_array)


class TestRemoteArrayBinaryOperations(MpiTestCase):

    @comm_null_passes
    def check_op(self, op):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a RemoteArray.
        """
        x1 = da.ones((16, 16), dist=('b', None), comm=self.comm)
        x2 = da.ones((16, 16), dist=('b', None), comm=self.comm)
        y = da.ones((16, 16), dist=('b', None), comm=self.comm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result0 = op(x1, x2, casting='unsafe')  # standard form
            op(x1, x2, y=y, casting='unsafe')  # three-arg form
        assert_array_equal(result0.remote_array, y.remote_array)


uops = (dc.absolute, dc.arccos, dc.arccosh, dc.arcsin, dc.arcsinh, dc.arctan,
        dc.arctanh, dc.conjugate, dc.cos, dc.cosh, dc.exp, dc.expm1, dc.log,
        dc.log10, dc.log1p, dc.negative, dc.reciprocal, dc.rint, dc.sign,
        dc.sin, dc.sinh, dc.sqrt, dc.square, dc.tan, dc.tanh, dc.invert,)

bops = (dc.add, dc.arctan2, dc.divide, dc.floor_divide, dc.fmod, dc.hypot,
        dc.mod, dc.multiply, dc.power, dc.remainder, dc.subtract,
        dc.true_divide, dc.less, dc.less_equal, dc.equal, dc.not_equal,
        dc.greater, dc.greater_equal, dc.bitwise_and, dc.bitwise_or,
        dc.bitwise_xor, dc.left_shift, dc.right_shift,)


# These operations don't work with our default data.
broken_tests = (dc.invert, dc.bitwise_and, dc.bitwise_or, dc.bitwise_xor,
                dc.left_shift, dc.right_shift,)

add_checkers(TestRemoteArrayUnaryOperations, uops, broken_tests)
add_checkers(TestRemoteArrayBinaryOperations, bops, broken_tests)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
