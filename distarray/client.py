# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import numpy as np
from distarray.externals.six import next

from itertools import product

from distarray.utils import has_exactly_one

__all__ = ['DistArray']


#----------------------------------------------------------------------------
# Code
#----------------------------------------------------------------------------

def process_return_value(subcontext, result_key):
    """Figure out what to return on the Client.

    Parameters
    ----------
    key : string
        Key corresponding to wrapped function's return value.

    Returns
    -------
    A DistArray (if locally all values are DistArray), a None (if
    locally all values are None), or else, pull the result back to the
    client and return it.  If all but one of the pulled values is None,
    return that non-None value only.
    """
    type_key = subcontext._generate_key()
    type_statement = "{} = str(type({}))".format(type_key, result_key)
    subcontext._execute(type_statement)
    result_type_str = subcontext._pull(type_key)

    def is_NoneType(typestring):
        return (typestring == "<type 'NoneType'>" or
                typestring == "<class 'NoneType'>")

    def is_LocalArray(typestring):
        return typestring == "<class 'distarray.local.denselocalarray.DenseLocalArray'>"

    if all(is_LocalArray(r) for r in result_type_str):
        result = DistArray(result_key, subcontext)
    elif all(is_NoneType(r) for r in result_type_str):
        result = None
    else:
        result = subcontext._pull(result_key)
        if has_exactly_one(result):
            result = next(x for x in result if x is not None)

    return result


class DistArray(object):

    __array_priority__ = 20.0

    def __init__(self, key, context):
        self.key = key
        self.context = context

    def __del__(self):
        self.context._execute('del %s' % self.key)

    def _get_attribute(self, name):
        key = self.context._generate_key()
        self.context._execute0('%s = %s.%s' % (key, self.key, name))
        result = self.context._pull0(key)
        return result

    def __repr__(self):
        s = '<DistArray(shape=%r, targets=%r)>' % \
            (self.shape, self.context.targets)
        return s

    def __getitem__(self, index):
        #TODO: FIXME: major performance improvements possible here,
        # especially for special casese like `index == slice(None)`.
        # This would dramatically improve tondarray's performance.

        if isinstance(index, int) or isinstance(index, slice):
            tuple_index = (index,)
            return self.__getitem__(tuple_index)

        elif isinstance(index, tuple):
            result_key = self.context._generate_key()
            fmt = '%s = %s.checked_getitem(%s)'
            statement = fmt % (result_key, self.key, index)
            self.context._execute(statement)
            result = process_return_value(self.context, result_key)
            if result is None:
                raise IndexError
            else:
                return result

        else:
            raise TypeError("Invalid index type.")

    def __setitem__(self, index, value):
        #TODO: FIXME: major performance improvements possible here.
        # Especially when `index == slice(None)` and value is an
        # ndarray, since for block and cyclic, we can generate slices of
        # `value` and assign to local arrays. This would dramatically
        # improve the fromndarray method's performance.

        if isinstance(index, int) or isinstance(index, slice):
            tuple_index = (index,)
            return self.__setitem__(tuple_index, value)

        elif isinstance(index, tuple):
            result_key = self.context._generate_key()
            fmt = '%s = %s.checked_setitem(%s, %s)'
            statement = fmt % (result_key, self.key, index, value)
            self.context._execute(statement)
            result = process_return_value(self.context, result_key)
            if result is None:
                raise IndexError()

        else:
            raise TypeError("Invalid index type.")

    @property
    def shape(self):
        return self._get_attribute('shape')

    @property
    def size(self):
        return self._get_attribute('size')

    @property
    def dist(self):
        return self._get_attribute('dist')

    @property
    def dtype(self):
        return self._get_attribute('dtype')

    @property
    def grid_shape(self):
        return self._get_attribute('grid_shape')

    @property
    def ndim(self):
        return self._get_attribute('ndim')

    @property
    def nbytes(self):
        return self._get_attribute('nbytes')

    @property
    def item_size(self):
        return self._get_attribute('item_size')

    def tondarray(self):
        """Returns the distributed array as an ndarray."""
        arr = np.empty(self.shape, dtype=self.dtype)
        local_name = self.context._generate_key()
        self.context._execute('%s = %s.copy()' % (local_name, self.key))
        local_arrays = self.context._pull(local_name)
        for local_array in local_arrays:
            maps = (ax_map.global_index for ax_map in local_array.maps)
            for index in product(*maps):
                arr[index] = local_array[index]
        return arr

    toarray = tondarray

    def get_dist_matrix(self):
        key = self.context._generate_key()
        self.context._execute0(
            '%s = %s.get_dist_matrix()' % (key, self.key))
        result = self.context._pull0(key)
        return result

    def fill(self, value):
        value_key = self.context._generate_key()
        self.context._push({value_key:value})
        self.context._execute('%s.fill(%s)' % (self.key, value_key))

    def sum(self, axis=None, dtype=None, out=None):
        keys = self.context._key_and_push(axis, dtype)
        result_key = self.context._generate_key()
        subs = (result_key, self.key) + keys
        self.context._execute('%s = %s.sum(%s,%s)' % subs)
        result = self.context._pull0(result_key)
        return result

    def mean(self, axis=None, dtype=None, out=None):
        keys = self.context._key_and_push(axis, dtype)
        result_key = self.context._generate_key()
        subs = (result_key, self.key) + keys
        self.context._execute('%s = %s.mean(%s,%s)' % subs)
        result = self.context._pull0(result_key)
        return result

    def var(self, axis=None, dtype=None, out=None):
        keys = self.context._key_and_push(axis, dtype)
        result_key = self.context._generate_key()
        subs = (result_key, self.key) + keys
        self.context._execute('%s = %s.var(%s,%s)' % subs)
        result = self.context._pull0(result_key)
        return result

    def std(self, axis=None, dtype=None, out=None):
        keys = self.context._key_and_push(axis, dtype)
        result_key = self.context._generate_key()
        subs = (result_key, self.key) + keys
        self.context._execute('%s = %s.std(%s,%s)' % subs)
        result = self.context._pull0(result_key)
        return result

    def get_ndarrays(self):
        """Pull the local ndarrays from the engines.

        Returns
        -------
        list of ndarrays
            one ndarray per process

        """
        key = self.context._generate_key()
        self.context._execute('%s = %s.get_localarray()' % (key, self.key))
        result = self.context._pull(key)
        return result

    def get_localarrays(self):
        """Pull the LocalArray objects from the engines.

        Returns
        -------
        list of localarrays
            one localarray per process

        """
        result = self.context._pull(self.key)
        return result

    def get_localshapes(self):
        key = self.context._generate_key()
        self.context._execute('%s = %s.local_shape' % (key, self.key))
        result = self.context._pull(key)
        return result

    # Binary operators

    def _binary_op_from_ufunc(self, other, func, rop_str=None, *args, **kwargs):
        if hasattr(other, '__array_priority__') and hasattr(other, rop_str):
            if other.__array_priority__ > self.__array_priority__:
                rop = getattr(other, rop_str)
                return rop(self)
        return func(self, other, *args, **kwargs)

    def _rbinary_op_from_ufunc(self, other, func, lop_str, *args, **kwargs):
        if hasattr(other, '__array_priority__') and hasattr(other, lop_str):
            if other.__array_priority__ > self.__array_priority__:
                lop = getattr(other, lop_str)
                return lop(self)
        return func(other, self, *args, **kwargs)

    def __add__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.add, '__radd__', *args, **kwargs)

    def __sub__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.subtract, '__rsub__', *args, **kwargs)

    def __mul__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.multiply, '__rmul__', *args, **kwargs)

    def __div__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.divide, '__rdiv__', *args, **kwargs)

    def __truediv__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.true_divide, '__rtruediv__', *args, **kwargs)

    def __floordiv__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.floor_divide, '__rfloordiv__', *args, **kwargs)

    def __mod__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.mod, '__rdiv__', *args, **kwargs)

    def __pow__(self, other, modulo=None, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.power, '__rpower__', *args, **kwargs)

    def __lshift__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.left_shift, '__rlshift__', *args, **kwargs)

    def __rshift__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.right_shift, '__rrshift__', *args, **kwargs)

    def __and__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.bitwise_and, '__rand__', *args, **kwargs)

    def __or__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.bitwise_or, '__ror__', *args, **kwargs)

    def __xor__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.bitwise_xor, '__rxor__', *args, **kwargs)

    # Binary - right versions

    def __radd__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.add, '__add__', *args, **kwargs)

    def __rsub__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.subtract, '__sub__', *args, **kwargs)

    def __rmul__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.multiply, '__mul__', *args, **kwargs)

    def __rdiv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.divide, '__div__', *args, **kwargs)

    def __rtruediv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.true_divide, '__truediv__', *args, **kwargs)

    def __rfloordiv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.floor_divide, '__floordiv__', *args, **kwargs)

    def __rmod__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.mod, '__mod__', *args, **kwargs)

    def __rpow__(self, other, modulo=None, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.power, '__pow__', *args, **kwargs)

    def __rlshift__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.left_shift, '__lshift__', *args, **kwargs)

    def __rrshift__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.right_shift, '__rshift__', *args, **kwargs)

    def __rand__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_and, '__and__', *args, **kwargs)

    def __ror__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_or, '__or__', *args, **kwargs)

    def __rxor__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_xor, '__xor__', *args, **kwargs)

    def __neg__(self, *args, **kwargs):
        return self.context.negative(self, *args, **kwargs)

    def __pos__(self, *args, **kwargs):
        return self

    def __abs__(self, *args, **kwargs):
        return self.context.abs(self, *args, **kwargs)

    def __invert__(self, *args, **kwargs):
        return self.context.invert(self, *args, **kwargs)

    # Boolean comparisons

    def __lt__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.less, '__lt__', *args, **kwargs)

    def __le__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.less_equal, '__le__', *args, **kwargs)

    def __eq__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.equal, '__eq__', *args, **kwargs)

    def __ne__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.not_equal, '__ne__', *args, **kwargs)

    def __gt__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.greater, '__gt__', *args, **kwargs)

    def __ge__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, self.context.greater_equal, '__ge__', *args, **kwargs)
