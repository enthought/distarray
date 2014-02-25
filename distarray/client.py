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
from itertools import product

import numpy as np
from six import next

from IPython.parallel import Client
from distarray.utils import has_exactly_one

__all__ = ['DistArray', 'Context']


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


class Context(object):

    def __init__(self, view=None, targets=None):
        if view is None:
            c = Client()
            self.view = c[:]
        else:
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
        self.view.execute("import distarray.local; import distarray.mpiutils")

        self._make_intracomm()
        self._set_engine_rank_mapping()

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
            '%s = distarray.mpiutils.create_comm_with_list(%s)' % (self._comm_key, ranks),
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
            '%s = distarray.local.zeros(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def ones(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        keys = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        subs = (da_key,) + keys + (self._comm_key,)
        self._execute(
            '%s = distarray.local.ones(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def empty(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        keys = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        subs = (da_key,) + keys + (self._comm_key,)
        self._execute(
            '%s = distarray.local.empty(%s, %s, %s, %s, %s)' % subs
        )
        return DistArray(da_key, self)

    def save(self, filename, da):
        """
        Save a distributed array to files in the ``.dnpy`` format.

        Parameters
        ----------
        filename : str
            Prefix for filename used by each engine.  Each engine will save a
            file named ``<filename>_<comm_rank>.dnpy``.
        da : DistArray
            Array to save to files.

        """
        subs = self._key_and_push(filename) + (da.key,)
        self._execute(
            'distarray.local.save(%s, %s)' % subs
        )

    def load(self, filename):
        """
        Load a distributed array from ``.dnpy`` files.

        Parameters
        ----------
        filename : str
            Prefix used for the file saved by each engine.  Each engine will
            load a file named ``<filename>_<comm_rank>.dnpy``.

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded on each engine.

        """
        da_key = self._generate_key()
        subs = (da_key, filename, self._comm_key)
        self._execute(
            '%s = distarray.local.load("%s", comm=%s)' % subs
        )
        return DistArray(da_key, self)

    def fromndarray(self, arr, dist={0: 'b'}, grid_shape=None):
        """Convert an ndarray to a distarray."""
        out = self.empty(arr.shape, dtype=arr.dtype, dist=dist,
                         grid_shape=grid_shape)
        for index, value in np.ndenumerate(arr):
            out[index] = value
        return out

    fromarray = fromndarray

    def fromfunction(self, function, shape, **kwargs):
        func_key = self._generate_key()
        self.view.push_function({func_key:function},targets=self.targets,block=True)
        keys = self._key_and_push(shape, kwargs)
        new_key = self._generate_key()
        subs = (new_key,func_key) + keys
        self._execute('%s = distarray.local.fromfunction(%s,%s,**%s)' % subs)
        return DistArray(new_key, self)

    def negative(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'negative', *args, **kwargs)
    def absolute(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'absolute', *args, **kwargs)
    def rint(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'rint', *args, **kwargs)
    def sign(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'sign', *args, **kwargs)
    def conjugate(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'conjugate', *args, **kwargs)
    def exp(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'exp', *args, **kwargs)
    def log(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'log', *args, **kwargs)
    def expm1(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'expm1', *args, **kwargs)
    def log1p(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'log1p', *args, **kwargs)
    def log10(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'log10', *args, **kwargs)
    def sqrt(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'sqrt', *args, **kwargs)
    def square(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'square', *args, **kwargs)
    def reciprocal(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'reciprocal', *args, **kwargs)
    def sin(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'sin', *args, **kwargs)
    def cos(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'cos', *args, **kwargs)
    def tan(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'tan', *args, **kwargs)
    def arcsin(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arcsin', *args, **kwargs)
    def arccos(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arccos', *args, **kwargs)
    def arctan(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arctan', *args, **kwargs)
    def sinh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'sinh', *args, **kwargs)
    def cosh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'cosh', *args, **kwargs)
    def tanh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'tanh', *args, **kwargs)
    def arcsinh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arcsinh', *args, **kwargs)
    def arccosh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arccosh', *args, **kwargs)
    def arctanh(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'arctanh', *args, **kwargs)
    def invert(self, a, *args, **kwargs):
        return unary_proxy(self, a, 'invert', *args, **kwargs)

    def add(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'add', *args, **kwargs)
    def subtract(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'subtract', *args, **kwargs)
    def multiply(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'multiply', *args, **kwargs)
    def divide(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'divide', *args, **kwargs)
    def true_divide(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'true_divide', *args, **kwargs)
    def floor_divide(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'floor_divide', *args, **kwargs)
    def power(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'power', *args, **kwargs)
    def remainder(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'remainder', *args, **kwargs)
    def fmod(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'fmod', *args, **kwargs)
    def arctan2(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'arctan2', *args, **kwargs)
    def hypot(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'hypot', *args, **kwargs)
    def bitwise_and(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'bitwise_and', *args, **kwargs)
    def bitwise_or(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'bitwise_or', *args, **kwargs)
    def bitwise_xor(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'bitwise_xor', *args, **kwargs)
    def left_shift(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'left_shift', *args, **kwargs)
    def right_shift(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'right_shift', *args, **kwargs)

    def mod(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'mod', *args, **kwargs)
    def rmod(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'rmod', *args, **kwargs)

    def less(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'less', *args, **kwargs)
    def less_equal(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'less_equal', *args, **kwargs)
    def equal(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'equal', *args, **kwargs)
    def not_equal(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'not_equal', *args, **kwargs)
    def greater(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'greater', *args, **kwargs)
    def greater_equal(self, a, b, *args, **kwargs):
        return binary_proxy(self, a, b, 'greater_equal', *args, **kwargs)

def unary_proxy(context, a, meth_name, *args, **kwargs):
    if not isinstance(a, DistArray):
        raise TypeError("This method only works on DistArrays")
    if  context != a.context:
        raise TypeError("distarray context mismatch: " % (context,
                                                          a.context))
    context = a.context
    new_key = context._generate_key()
    if 'casting' in kwargs:
        exec_str = "%s = distarray.local.%s(%s, casting='%s')" % (
                new_key, meth_name, a.key, kwargs['casting'],
                )
    else:
        exec_str = '%s = distarray.local.%s(%s)' % (
                new_key, meth_name, a.key,
                )
    context._execute(exec_str)
    return DistArray(new_key, context)

def binary_proxy(context, a, b, meth_name, *args, **kwargs):
    is_a_dap = isinstance(a, DistArray)
    is_b_dap = isinstance(b, DistArray)
    if is_a_dap and is_b_dap:
        if b.context != a.context:
            raise TypeError("distarray context mismatch: " % (b.context,
                                                              a.context))
        if context != a.context:
            raise TypeError("distarray context mismatch: " % (context,
                                                              a.context))
        a_key = a.key
        b_key = b.key
    elif is_a_dap and np.isscalar(b):
        if context != a.context:
            raise TypeError("distarray context mismatch: " % (context,
                                                              a.context))
        a_key = a.key
        b_key = context._key_and_push(b)[0]
    elif is_b_dap and np.isscalar(a):
        if context != b.context:
            raise TypeError("distarray context mismatch: " % (context,
                                                              b.context))
        a_key = context._key_and_push(a)[0]
        b_key = b.key
    else:
        raise TypeError('only DistArray or scalars are accepted')
    new_key = context._generate_key()

    if 'casting' in kwargs:
        exec_str = "%s = distarray.local.%s(%s,%s, casting='%s')" % (
                new_key, meth_name, a_key, b_key, kwargs['casting'],
                )
    else:
        exec_str = '%s = distarray.local.%s(%s,%s)' % (
                new_key, meth_name, a_key, b_key,
                )

    context._execute(exec_str)
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
