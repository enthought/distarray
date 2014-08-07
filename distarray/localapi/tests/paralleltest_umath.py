# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import warnings
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import distarray.localapi as dc
import distarray.localapi.localarray as localarray
from distarray.localapi.localarray import LocalArray
from distarray.localapi.maps import Distribution
from distarray.localapi.error import IncompatibleArrayError
from distarray.testing import ParallelTestCase


class TestUnaryUFunc(ParallelTestCase):

    def test_negative(self):
        """See if unary ufunc works for a LocalArray."""
        d = Distribution.from_shape(comm=self.comm, shape=(16, 16))
        a = LocalArray(d, dtype='int32')
        a.fill(1)
        b = localarray.negative(a)
        self.assertTrue(np.all(a.ndarray == -b.ndarray))

        b = localarray.empty_like(a)
        b = localarray.negative(a, b)
        self.assertTrue(np.all(a.ndarray == -b.ndarray))

        d2 = Distribution.from_shape(comm=self.comm, shape=(20, 20))
        a = LocalArray(d, dtype='int32')
        b = LocalArray(d2, dtype='int32')
        self.assertRaises(IncompatibleArrayError, localarray.negative, b, a)

    def test_abs(self):
        """See if unary ufunc works for a LocalArray."""
        d = Distribution.from_shape(comm=self.comm, shape=(16, 16))
        a = LocalArray(d, dtype='int32')
        a.fill(-5)
        a[2, 3] = 11
        b = abs(a)
        self.assertTrue(np.all(abs(a.ndarray) == b.ndarray))


class TestBinaryUFunc(ParallelTestCase):

    def test_add(self):
        """See if binary ufunc works for a LocalArray."""
        d = Distribution.from_shape(comm=self.comm, shape=(16, 16))
        a = LocalArray(d, dtype='int32')
        b = LocalArray(d, dtype='int32')
        a.fill(1)
        b.fill(1)
        c = localarray.add(a, b)
        self.assertTrue(np.all(c.ndarray == 2))

        c = localarray.empty_like(a)
        c = localarray.add(a, b, c)
        self.assertTrue(np.all(c.ndarray == 2))

        d0 = Distribution.from_shape(comm=self.comm, shape=(16, 16))
        d1 = Distribution.from_shape(comm=self.comm, shape=(20, 20))
        a = LocalArray(d0, dtype='int32')
        b = LocalArray(d1, dtype='int32')
        self.assertRaises(IncompatibleArrayError, localarray.add, a, b)

        d0 = Distribution.from_shape(comm=self.comm, shape=(16, 16))
        d1 = Distribution.from_shape(comm=self.comm, shape=(20, 20))
        a = LocalArray(d0, dtype='int32')
        b = LocalArray(d0, dtype='int32')
        c = LocalArray(d1, dtype='int32')
        self.assertRaises(IncompatibleArrayError, localarray.add, a, b, c)


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


class TestLocalArrayUnaryOperations(ParallelTestCase):

    def check_op(self, op):
        """Check unary operation for success.

        Check the one- and two-arg ufunc versions as well as the method
        version attached to a LocalArray.
        """
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), dist=('b', 'n'))
        x = localarray.ones(d)
        y = localarray.ones(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result0 = op(x, casting='unsafe')  # standard form
            op(x, y=y, casting='unsafe')  # two-arg form
        assert_array_equal(result0.ndarray, y.ndarray)


class TestLocalArrayBinaryOperations(ParallelTestCase):

    def check_op(self, op):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), dist=('b', 'n'))
        x1 = localarray.ones(d)
        x2 = localarray.ones(d)
        y = localarray.ones(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result0 = op(x1, x2, casting='unsafe')  # standard form
            op(x1, x2, y=y, casting='unsafe')  # three-arg form
        assert_array_equal(result0.ndarray, y.ndarray)


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

add_checkers(TestLocalArrayUnaryOperations, uops, broken_tests)
add_checkers(TestLocalArrayBinaryOperations, bops, broken_tests)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
