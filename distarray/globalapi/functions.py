# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Distributed ufuncs for `DistArray`\s.
"""

from __future__ import absolute_import

import numpy

from distarray.error import ContextError
from distarray.globalapi.distarray import DistArray


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

        def func_call(func_name, arr_name, args, kwargs):
            from distarray.utils import get_from_dotted_name
            dotted_name = 'distarray.localapi.%s' % (func_name,)
            func = get_from_dotted_name(dotted_name)
            res = func(arr_name, *args, **kwargs)
            return proxyize(res), res.dtype  # noqa

        res = context.apply(func_call, args=(name, a.key, args, kwargs),
                            targets=a.targets)
        new_key = res[0][0]
        dtype = res[0][1]
        return DistArray.from_localarrays(new_key,
                                          distribution=a.distribution,
                                          dtype=dtype)
    return proxy_func


def binary_proxy(name):
    def proxy_func(a, b, *args, **kwargs):
        context = determine_context(a, b)
        is_a_dap = isinstance(a, DistArray)
        is_b_dap = isinstance(b, DistArray)
        if is_a_dap and is_b_dap:
            if not a.distribution.is_compatible(b.distribution):
                raise ValueError("distributions not compatible.")
            a_key = a.key
            b_key = b.key
            distribution = a.distribution
        elif is_a_dap and numpy.isscalar(b):
            a_key = a.key
            b_key = b
            distribution = a.distribution
        elif is_b_dap and numpy.isscalar(a):
            a_key = a
            b_key = b.key
            distribution = b.distribution
        else:
            raise TypeError('only DistArray or scalars are accepted')

        def func_call(func_name, a, b, args, kwargs):
            from distarray.utils import get_from_dotted_name
            dotted_name = 'distarray.localapi.%s' % (func_name,)
            func = get_from_dotted_name(dotted_name)
            res = func(a, b, *args, **kwargs)
            return proxyize(res), res.dtype  # noqa

        res = context.apply(func_call, args=(name, a_key, b_key, args, kwargs),
                            targets=distribution.targets)
        new_key = res[0][0]
        dtype = res[0][1]
        return DistArray.from_localarrays(new_key,
                                          distribution=distribution,
                                          dtype=dtype)
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
