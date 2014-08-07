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

import numpy as np
from numpy.testing import assert_allclose

from distarray.testing import DefaultContextTestCase
import distarray.globalapi.functions as functions
from distarray.globalapi import Context

SKIP = True
try:
    glb_ctx = Context()
    SKIP = False
except EnvironmentError:
    pass

def setUpModule():
    global glb_ctx
    if not SKIP and not glb_ctx:
        glb_ctx = Context()

def tearDownModule():
    if not SKIP and glb_ctx:
        glb_ctx.close()

def add_checkers(cls, ops_and_data, checker_name):
    """Helper function to dynamically add a list of tests.

    Add tests to cls for each op in ops. Where checker_name is
    the name of the test you want to call on each op. So we add:

        TestCls.test_op_name(): return op_checker(op_name)

    for each op.
    """
    op_checker = getattr(cls, checker_name)

    ops, data = ops_and_data

    if not SKIP:
        dist_data = tuple(glb_ctx.fromndarray(d) for d in data)

    def check(op_name):
        if SKIP:
            return lambda self: None
        else:
            return lambda self: op_checker(self, op_name, data, dist_data)

    for op_name in ops:
        op_test_name = 'test_' + op_name
        setattr(cls, op_test_name, check(op_name))


class TestDistArrayUfuncs(DefaultContextTestCase):
    """Test ufuncs operating on distarrays"""

    ntargets = 'any'

    def check_binary_op(self, op_name, data, dist_data):
        """Check binary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        op = getattr(functions, op_name)
        ufunc = getattr(np, op_name)
        a, b = data
        da, db = dist_data
        expected = ufunc(a, b, casting='unsafe')
        result = op(da, db, casting='unsafe')
        assert_allclose(result.toarray(), expected)

    def check_unary_op(self, op_name, data, dist_data):
        """Check unary operation for success.

        Check the two- and three-arg ufunc versions as well as the
        method version attached to a LocalArray.
        """
        op = getattr(functions, op_name)
        ufunc = getattr(np, op_name)
        a, = data
        da, = dist_data
        expected = ufunc(a, casting='unsafe')
        result = op(da, casting='unsafe')
        assert_allclose(result.toarray(), expected)


class TestSpecialMethods(DefaultContextTestCase):
    """Test the __methods__"""

    ntargets = 'any'

    def check_binary_op(self, op_name, data, dist_data):
        a, b = data
        da, db = dist_data
        distop = getattr(da, op_name)
        numpyop = getattr(a, op_name)
        result = distop(db)
        expected = numpyop(b)
        assert_allclose(result.toarray(), expected)

    def check_unary_op(self, op_name, data, dist_data):
        a, = data
        da, = dist_data
        distop = getattr(da, op_name)
        numpyop = getattr(a, op_name)
        result = distop()
        expected = numpyop()
        assert_allclose(result.toarray(), expected)


arr_a = np.arange(1, 11)
arr_b = np.ones_like(arr_a) * 2
arr_c = np.random.rand(100)

unary_ops_1 = (('absolute', 'arccosh', 'arcsinh', 'conjugate', 'cos',
'cosh', 'exp', 'expm1', 'log', 'log10', 'log1p', 'negative', 'reciprocal',
'rint', 'sign', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'invert'),
(arr_a,))

unary_ops_2 = (('arccos', 'arcsin', 'arctanh'), (arr_c,))

unary_special_methods = (('__neg__', '__pos__', '__abs__', '__invert__'),
(arr_a,))

binary_ops = (('add', 'arctan2', 'divide', 'floor_divide', 'fmod', 'hypot',
              'multiply', 'power', 'remainder', 'subtract', 'true_divide',
              'less', 'less_equal', 'equal', 'not_equal', 'greater',
              'greater_equal', 'mod', 'bitwise_and', 'bitwise_or',
              'bitwise_xor', 'left_shift', 'right_shift',),
              (arr_a, arr_b))

binary_special_methods = (('__lt__', '__le__', '__eq__', '__ne__', '__gt__',
                          '__ge__', '__add__', '__sub__', '__mul__',
                          '__floordiv__', '__mod__', '__pow__', '__radd__',
                          '__rsub__', '__rmul__', '__rfloordiv__', '__rmod__',
                          '__rpow__', '__rrshift__', '__rlshift__',
                          '__rand__', '__rxor__', '__ror__', '__lshift__',
                          '__rshift__', '__and__', '__xor__', '__or__',),
                          (arr_a, arr_b))

# There is no divmod function in numpy. And there is no __div__
# attribute on ndarrays.
problematic_special_methods = ('__divmod__', '__rdivmod__', '__div__')

add_checkers(TestDistArrayUfuncs, binary_ops, 'check_binary_op')
add_checkers(TestSpecialMethods, binary_special_methods, 'check_binary_op')
add_checkers(TestDistArrayUfuncs, unary_ops_1, 'check_unary_op')
add_checkers(TestDistArrayUfuncs, unary_ops_2, 'check_unary_op')
add_checkers(TestSpecialMethods, unary_special_methods, 'check_unary_op')


if __name__ == '__main__':
    unittest.main(verbosity=2)
