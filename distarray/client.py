# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import uuid
import numpy as np
from six import next

from distarray.utils import has_exactly_one


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


class RandomModule(object):

    def __init__(self, context):
        self.context = context
        self.context._execute('import distarray.random')

    def rand(self, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.rand(%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def normal(self, loc=0.0, scale=1.0, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(loc, scale, size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.normal(%s,%s,%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def randint(self, low, high=None, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(low, high, size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.randint(%s,%s,%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def randn(self, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.randn(%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)


class Context(object):

    def __init__(self, view, targets=None):
        self.view = view

        all_targets = self.view.targets
        if targets is None:
            self.targets = all_targets
        else:
            self.targets = []
            for target in targets:
                assert target in all_targets, "Engine with id %r not registered" % target
                self.targets.append(target)

        # FIXME: IPython bug #4296: This doesn't work under Python 3
        #with self.view.sync_imports():
        #    import distarray
        self.view.execute("import distarray")

        self._make_intracomm()
        self._set_engine_rank_mapping()

        # self.random = RandomModule(self)

    def _set_engine_rank_mapping(self):
        # The MPI intracomm referred to by self._comm_key may have a different
        # mapping between IPython engines and MPI ranks than COMM_PRIVATE.  Set
        # self.ranks to this mapping.
        rank = self._generate_key()
        self.view.execute(
                '%s = %s.Get_rank()' % (rank, self._comm_key),
                block=True, targets=self.targets)
        self.target_to_rank = self.view.pull(rank, targets=self.targets).get_dict()

        # ensure consistency
        assert set(self.targets) == set(self.target_to_rank.keys())
        assert set(range(len(self.targets))) == set(self.target_to_rank.values())

    def _make_intracomm(self):
        def get_rank():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_rank()

        # get a mapping of IPython engine ID to MPI rank
        rank_map = self.view.apply_async(get_rank).get_dict()
        ranks = [ rank_map[engine] for engine in self.targets ]

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_size()

        comm_size = self.view.apply_async(get_size).get()[0]
        if set(rank_map.values()) != set(range(comm_size)):
            raise ValueError('Engines in view must encompass all MPI ranks.')

        # create a new communicator with the subset of engines note that
        # MPI_Comm_create must be called on all engines, not just those
        # involved in the new communicator.
        self._comm_key = self._generate_key()
        self.view.execute(
            '%s = distarray.create_comm_with_list(%s)' % (self._comm_key, ranks),
            block=True
        )

    def _generate_key(self):
        uid = uuid.uuid4()
        return '__distarray_%s' % uid.hex

    def _key_and_push(self, *values):
        keys = [self._generate_key() for value in values]
        self._push(dict(zip(keys, values)))
        return tuple(keys)

    def _execute(self, lines, targets=None):
        if targets is None:
            targets = self.targets
        return self.view.execute(lines,targets=targets,block=True)

    def _push(self, d, targets=None):
        if targets is None:
            targets = self.targets
        return self.view.push(d,targets=targets,block=True)

    def _pull(self, k, targets=None):
        if targets is None:
            targets = self.targets
        return self.view.pull(k,targets=targets,block=True)

    def _execute0(self, lines):
        return self.view.execute(lines,targets=self.targets[0],block=True)

    def _push0(self, d):
        return self.view.push(d,targets=self.targets[0],block=True)

    def _pull0(self, k):
        return self.view.pull(k,targets=self.targets[0],block=True)

    def zeros(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        keys = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        subs = (da_key,) + keys + (self._comm_key,)
        self._execute(
            '%s = distarray.zeros(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def ones(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        keys = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        subs = (da_key,) + keys + (self._comm_key,)
        self._execute(
            '%s = distarray.ones(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def empty(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        keys = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        subs = (da_key,) + keys + (self._comm_key,)
        self._execute(
            '%s = distarray.empty(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def save(self, filename, da):
        self._execute("import pickle")
        self._push({'filename': filename})
        self._execute(
            'local_filename = filename + "_" + str(%s.comm_rank) + ".dar"' % da.key
        )
        self._execute(
            'pickle.dump(%s.__distarray__(), open(local_filename, "wb"))' % da.key
        )

    def load(self, filename):
        self._execute("import pickle")
        self._push({'filename': filename})
        distbuffer_key = self._generate_key()
        da_key = self._generate_key()
        self._execute(
            'local_filename = filename + "_" + str(%s.Get_rank()) + ".dar"' % self._comm_key
        )
        result = self._pull('local_filename')
        self._execute(
            '%s = pickle.load(open(local_filename, "rb"))' % distbuffer_key
        )
        self._execute(
            '%s = distarray.denselocalarray.LocalArray.from_distarray(%s)' % (da_key, distbuffer_key)
        )
        return DistArray(da_key, self)

    def fromndarray(self, arr):
        keys = self._key_and_push(arr.shape, arr.dtype)
        arr_key = self._generate_key()
        new_key = self._generate_key()
        self.view.scatter(arr_key, arr, targets=self.targets, block=True)
        subs = (new_key,) + keys + (arr_key,)
        self._execute(
            '%s = distarray.LocalArray(%s,dtype=%s,buf=%s)' % subs
        )
        return DistArray(new_key, self)

    fromarray = fromndarray

    def fromfunction(self, function, shape, **kwargs):
        func_key = self._generate_key()
        self.view.push_function({func_key:function},targets=self.targets,block=True)
        keys = self._key_and_push(shape, kwargs)
        new_key = self._generate_key()
        subs = (new_key,func_key) + keys
        self._execute('%s = distarray.fromfunction(%s,%s,**%s)' % subs)
        return DistArray(new_key, self)

    def negative(self, a): return unary_proxy(self, a, 'negative')
    def absolute(self, a): return unary_proxy(self, a, 'absolute')
    def rint(self, a): return unary_proxy(self, a, 'rint')
    def sign(self, a): return unary_proxy(self, a, 'sign')
    def conjugate(self, a): return unary_proxy(self, a, 'conjugate')
    def exp(self, a): return unary_proxy(self, a, 'exp')
    def log(self, a): return unary_proxy(self, a, 'log')
    def expm1(self, a): return unary_proxy(self, a, 'expm1')
    def log1p(self, a): return unary_proxy(self, a, 'log1p')
    def log10(self, a): return unary_proxy(self, a, 'log10')
    def sqrt(self, a): return unary_proxy(self, a, 'sqrt')
    def square(self, a): return unary_proxy(self, a, 'square')
    def reciprocal(self, a): return unary_proxy(self, a, 'reciprocal')
    def sin(self, a): return unary_proxy(self, a, 'sin')
    def cos(self, a): return unary_proxy(self, a, 'cos')
    def tan(self, a): return unary_proxy(self, a, 'tan')
    def arcsin(self, a): return unary_proxy(self, a, 'arcsin')
    def arccos(self, a): return unary_proxy(self, a, 'arccos')
    def arctan(self, a): return unary_proxy(self, a, 'arctan')
    def sinh(self, a): return unary_proxy(self, a, 'sinh')
    def cosh(self, a): return unary_proxy(self, a, 'cosh')
    def tanh(self, a): return unary_proxy(self, a, 'tanh')
    def arcsinh(self, a): return unary_proxy(self, a, 'arcsinh')
    def arccosh(self, a): return unary_proxy(self, a, 'arccosh')
    def arctanh(self, a): return unary_proxy(self, a, 'arctanh')
    def invert(self, a): return unary_proxy(self, a, 'invert')

    def add(self, a, b): return binary_proxy(self, a, b, 'add')
    def subtract(self, a, b): return binary_proxy(self, a, b, 'subtract')
    def multiply(self, a, b): return binary_proxy(self, a, b, 'multiply')
    def divide(self, a, b): return binary_proxy(self, a, b, 'divide')
    def true_divide(self, a, b): return binary_proxy(self, a, b, 'true_divide')
    def floor_divide(self, a, b): return binary_proxy(self, a, b, 'floor_divide')
    def power(self, a, b): return binary_proxy(self, a, b, 'power')
    def remainder(self, a, b): return binary_proxy(self, a, b, 'remainder')
    def fmod(self, a, b): return binary_proxy(self, a, b, 'fmod')
    def arctan2(self, a, b): return binary_proxy(self, a, b, 'arctan2')
    def hypot(self, a, b): return binary_proxy(self, a, b, 'hypot')
    def bitwise_and(self, a, b): return binary_proxy(self, a, b, 'bitwise_and')
    def bitwise_or(self, a, b): return binary_proxy(self, a, b, 'bitwise_or')
    def bitwise_xor(self, a, b): return binary_proxy(self, a, b, 'bitwise_xor')
    def left_shift(self, a, b): return binary_proxy(self, a, b, 'left_shift')
    def right_shift(self, a, b): return binary_proxy(self, a, b, 'right_shift')


def unary_proxy(context, a, meth_name):
    assert isinstance(a, DistArray), "This method only works on DistArrays"
    assert context==a.context, "distarray context mismatch: " % (context, a.context)
    context = a.context
    new_key = context._generate_key()
    context._execute('%s = distarray.%s(%s)' % (new_key, meth_name, a.key))
    return DistArray(new_key, context)

def binary_proxy(context, a, b, meth_name):
    is_a_dap = isinstance(a, DistArray)
    is_b_dap = isinstance(b, DistArray)
    if is_a_dap and is_b_dap:
        assert b.context==a.context, "distarray context mismatch: " % (b.context, a.context)
        assert context==a.context, "distarray context mismatch: " % (context, a.context)
        a_key = a.key
        b_key = b.key
    elif is_a_dap and np.isscalar(b):
        assert context==a.context, "distarray context mismatch: " % (context, a.context)
        a_key = a.key
        b_key = context._key_and_push(b)[0]
    elif is_b_dap and np.isscalar(a):
        assert context==b.context, "distarray context mismatch: " % (context, b.context)
        a_key = context._key_and_push(a)[0]
        b_key = b.key
    else:
        raise TypeError('only DistArray or scalars are accepted')
    new_key = context._generate_key()
    context._execute('%s = distarray.%s(%s,%s)' % (new_key, meth_name, a_key, b_key))
    return DistArray(new_key, context)


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
        local_name = self.context._generate_key()
        local_shape = self.context._generate_key()
        subs = (local_name, self.key, local_shape, self.key)
        self.context._execute('%s = %s.local_view(); %s = %s.shape' % subs)
        shape = self.context._pull0(local_shape)
        arr = self.context.view.gather(
            local_name,targets=self.context.targets,block=True)
        arr.shape = shape
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

    def get_localarrays(self):
        key = self.context._generate_key()
        self.context._execute('%s = %s.get_localarray()' % (key, self.key))
        result = self.context._pull(key)
        return result

    def get_localshapes(self):
        key = self.context._generate_key()
        self.context._execute('%s = %s.local_shape' % (key, self.key))
        result = self.context._pull(key)
        return result

    # Binary operators

    def _binary_op_from_ufunc(self, other, func, rop_str=None):
        if hasattr(other, '__array_priority__') and hasattr(other, rop_str):
            if other.__array_priority__ > self.__array_priority__:
                rop = getattr(other, rop_str)
                return rop(self)
        return func(self, other)

    def _rbinary_op_from_ufunc(self, other, func, lop_str):
        if hasattr(other, '__array_priority__') and hasattr(other, lop_str):
            if other.__array_priority__ > self.__array_priority__:
                lop = getattr(other, lop_str)
                return lop(self)
        return func(other, self)

    def __add__(self, other):
        return self._binary_op_from_ufunc(other, self.context.add, '__radd__')

    def __sub__(self, other):
        return self._binary_op_from_ufunc(other, self.context.subtract, '__rsub__')

    def __mul__(self, other):
        return self._binary_op_from_ufunc(other, self.context.multiply, '__rmul__')

    def __div__(self, other):
        return self._binary_op_from_ufunc(other, self.context.divide, '__rdiv__')

    def __truediv__(self, other):
        return self._binary_op_from_ufunc(other, self.context.true_divide, '__rtruediv__')

    def __floordiv__(self, other):
        return self._binary_op_from_ufunc(other, self.context.floor_divide, '__rfloordiv__')

    def __mod__(self, other):
        return self._binary_op_from_ufunc(other, self.context.mod, '__rdiv__')

    def __pow__(self, other, modulo=None):
        return self._binary_op_from_ufunc(other, self.context.power, '__rpower__')

    def __lshift__(self, other):
        return self._binary_op_from_ufunc(other, self.context.left_shift, '__rlshift__')

    def __rshift__(self, other):
        return self._binary_op_from_ufunc(other, self.context.right_shift, '__rrshift__')

    def __and__(self, other):
        return self._binary_op_from_ufunc(other, self.context.bitwise_and, '__rand__')

    def __or__(self, other):
        return self._binary_op_from_ufunc(other, self.context.binary_or, '__ror__')

    def __xor__(self, other):
        return self._binary_op_from_ufunc(other, self.context.binary_xor, '__rxor__')

    # Binary - right versions

    def __radd__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.add, '__add__')

    def __rsub__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.subtract, '__sub__')

    def __rmul__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.multiply, '__mul__')

    def __rdiv__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.divide, '__div__')

    def __rtruediv__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.true_divide, '__truediv__')

    def __rfloordiv__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.floor_divide, '__floordiv__')

    def __rmod__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.mod, '__mod__')

    def __rpow__(self, other, modulo=None):
        return self._rbinary_op_from_ufunc(other, self.context.power, '__pow__')

    def __rlshift__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.left_shift, '__lshift__')

    def __rrshift__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.right_shift, '__rshift__')

    def __rand__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_and, '__and__')

    def __ror__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_or, '__or__')

    def __rxor__(self, other):
        return self._rbinary_op_from_ufunc(other, self.context.bitwise_xor, '__xor__')

    def __neg__(self):
        return self.context.negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.context.abs(self)

    def __invert__(self):
        return self.context.invert(self)
