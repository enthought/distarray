# encoding: utf-8
# Copyright (c) 2008-2014, IPython Development Team and Enthought, Inc.
"""
Our adaptation of NumPy's ufuncs.
"""

import numpy

from distarray.error import ContextError
from distarray.client import DistArray


__docformat__ = "restructuredtext en"
__all__ = []  # unary_names and binary_names added to __all__ below.

# numpy unary operations to wrap
unary_names = ('absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
               'arctanh', 'conjugate', 'cos', 'cosh', 'exp', 'expm1', 'invert',
               'log', 'log10', 'log1p', 'negative', 'reciprocal', 'rint',
               'sign', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh')

# numpy binary operations to wrap
binary_names = ('add', 'arctan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
                'divide', 'floor_divide', 'fmod', 'hypot', 'left_shift', 'mod',
                'multiply', 'power', 'remainder', 'right_shift', 'subtract',
                'true_divide', 'less', 'less_equal', 'equal', 'not_equal',
                'greater', 'greater_equal',)

for func_name in unary_names + binary_names:
    __all__.append(func_name)


def unary_proxy(name):
    def proxy_func(a, *args, **kwargs):
        context = determine_context(a)
        new_key = context._generate_key()
        if 'casting' in kwargs:
            exec_str = "%s = distarray.local.%s(%s, casting='%s')"
            exec_str %= (new_key, name, a.key, kwargs['casting'])
        else:
            exec_str = '%s = distarray.local.%s(%s)'
            exec_str %= (new_key, name, a.key)

        context._execute(exec_str)
        return DistArray(new_key, context)
    return proxy_func


def binary_proxy(name):
    def proxy_func(a, b, *args, **kwargs):
        context = determine_context(a, b)
        is_a_dap = isinstance(a, DistArray)
        is_b_dap = isinstance(b, DistArray)
        if is_a_dap and is_b_dap:
            a_key = a.key
            b_key = b.key
        elif is_a_dap and numpy.isscalar(b):
            a_key = a.key
            b_key = context._key_and_push(b)[0]
        elif is_b_dap and numpy.isscalar(a):
            a_key = context._key_and_push(a)[0]
            b_key = b.key
        else:
            raise TypeError('only DistArray or scalars are accepted')
        new_key = context._generate_key()

        if 'casting' in kwargs:
            exec_str = "%s = distarray.local.%s(%s,%s, casting='%s')"
            exec_str %= (new_key, name, a_key, b_key, kwargs['casting'])
        else:
            exec_str = '%s = distarray.local.%s(%s,%s)'
            exec_str %= (new_key, name, a_key, b_key)

        context._execute(exec_str)
        return DistArray(new_key, context)
    return proxy_func


def determine_context(*args):
    """ Determine a context from a functions arguments."""

    contexts = []
    # inspect args for a context
    for arg in args:
        if isinstance(arg, DistArray):
            contexts.append(arg.context)

    # check the args had a context
    if contexts == []:
        raise TypeError('Function must take DistArray or Context objects.')

    # check that all contexts are equal
    if not contexts.count(contexts[0]) == len(contexts):
        msg = ("Arguments must use the same Context (given arguments of "
               "type %r)")
        msg %= (tuple(set(contexts)),)
        raise ContextError(msg)

    return contexts[0]

# Define the functions dynamically at the module level.
for name in unary_names:
    globals()[name] = unary_proxy(name)

for name in binary_names:
    globals()[name] = binary_proxy(name)
