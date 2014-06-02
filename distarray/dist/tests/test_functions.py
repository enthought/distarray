# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for distarray ufuncs.

Many of these tests require a 4-engine cluster to be running locally.
"""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from distarray.testing import ContextTestCase
import distarray.dist.functions as functions


def add_checkers(cls, ops, checker_name):
    """Helper function to dynamically add a list of tests.

    Add tests to cls for each op in ops. Where checker_name is
    the name of the test you want to call on each op. So we add:

        TestCls.test_op_name(): return op_checker(op_name)

    for each op.
    """
    op_checker = getattr(cls, checker_name)

    def check(op_name):
        return lambda self: op_checker(self, op_name)

    for op_name in ops:
        op_test_name = 'test_' + op_name
        setattr(cls, op_test_name, check(op_name))


class TestDistArrayUfuncs(ContextTestCase):
    """Test ufuncs operating on distarrays"""

    ntargets = 'any'

    @classmethod
    def setUpClass(cls):
        super(TestDistArrayUfuncs, cls).setUpClass()
        # Standard data
        cls.a = np.arange(1, 11)
        cls.b = np.ones_like(cls.a)*2
        # distributed array data
        cls.da = cls.context.fromndarray(cls.a)
        cls.db = cls.context.fromndarray(cls.b)

    def check_binary_op(self, op_name):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        op = getattr(functions, op_name)
        ufunc = getattr(np, op_name)
        with warnings.catch_warnings():
            # ignore inf, NaN warnings etc.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = ufunc(self.a, self.b, casting='unsafe')
            result = op(self.da, self.db, casting='unsafe')
        assert_allclose(result.toarray(), expected)

    def check_unary_op(self, op_name):
        """Check unary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        op = getattr(functions, op_name)
        ufunc = getattr(np, op_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = ufunc(self.a, casting='unsafe')
            result = op(self.da, casting='unsafe')
        assert_allclose(result.toarray(), expected)


class TestSpecialMethods(ContextTestCase):
    """Test the __methods__"""

    ntargets = 'any'

    @classmethod
    def setUpClass(cls):
        super(TestSpecialMethods, cls).setUpClass()
        # Standard data
        cls.a = np.arange(1, 11)
        cls.b = np.ones_like(cls.a)*2
        # distributed array data
        cls.da = cls.context.fromndarray(cls.a)
        cls.db = cls.context.fromndarray(cls.b)

    def check_op(self, op_name):
        distop = getattr(self.da, op_name)
        numpyop = getattr(self.a, op_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = distop(self.db)
            expected = numpyop(self.b)
        assert_allclose(result.toarray(), expected)


unary_ops = ('absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
             'arctanh', 'conjugate', 'cos', 'cosh', 'exp', 'expm1', 'log',
             'log10', 'log1p', 'negative', 'reciprocal', 'rint', 'sign', 'sin',
             'sinh', 'sqrt', 'square', 'tan', 'tanh', 'invert')

binary_ops = ('add', 'arctan2', 'divide', 'floor_divide', 'fmod', 'hypot',
              'multiply', 'power', 'remainder', 'subtract', 'true_divide',
              'less', 'less_equal', 'equal', 'not_equal', 'greater',
              'greater_equal', 'mod', 'bitwise_and', 'bitwise_or',
              'bitwise_xor', 'left_shift', 'right_shift',)

binary_special_methods = ('__lt__', '__le__', '__eq__', '__ne__', '__gt__',
                          '__ge__', '__add__', '__sub__', '__mul__',
                          '__floordiv__', '__mod__', '__pow__', '__radd__',
                          '__rsub__', '__rmul__', '__rfloordiv__', '__rmod__',
                          '__rpow__', '__rrshift__', '__rlshift__',
                          '__rand__', '__rxor__', '__ror__', '__lshift__',
                          '__rshift__', '__and__', '__xor__', '__or__',)

# There is no divmod function in numpy. And there is no __div__
# attribute on ndarrays.
problematic_special_methods = ('__divmod__', '__rdivmod__', '__div__')

add_checkers(TestDistArrayUfuncs, binary_ops, 'check_binary_op')
add_checkers(TestDistArrayUfuncs, unary_ops, 'check_unary_op')
add_checkers(TestSpecialMethods, binary_special_methods, 'check_op')


if __name__ == '__main__':
    unittest.main(verbosity=2)
