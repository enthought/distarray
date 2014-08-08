# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
The `DistArray` data structure.

`DistArray` objects are proxies for collections of `LocalArray` objects. They
are meant to roughly emulate NumPy `ndarray`\s.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import absolute_import, division

import operator
from itertools import product
from functools import reduce

import numpy as np

import distarray.localapi
from distarray.metadata_utils import sanitize_indices
from distarray.globalapi.maps import Distribution
from distarray.utils import _raise_nie
from distarray.metadata_utils import normalize_reduction_axes

__all__ = ['DistArray']


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

class DistArray(object):

    __array_priority__ = 20.0

    def __init__(self, distribution, dtype=float):
        """Creates an empty DistArray according to the `distribution` given."""

        def _local_create(comm, ddpr, dtype):
            from distarray.localapi import empty
            from distarray.localapi.maps import Distribution
            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
            dist = Distribution(comm=comm, dim_data=dim_data)
            return proxyize(empty(dist, dtype))

        ctx = distribution.context
        ddpr = distribution.get_dim_data_per_rank()

        da_key = ctx.apply(_local_create, (distribution.comm, ddpr, dtype),
                           targets=distribution.targets)

        self.distribution = distribution
        self.key = da_key[0]
        self._dtype = dtype

    @classmethod
    def from_localarrays(cls, key, context=None, targets=None, distribution=None,
                         dtype=None):
        """The caller has already created the LocalArray objects.  `key` is
        their name on the engines.  This classmethod creates a DistArray that
        refers to these LocalArrays.

        Either a `context` or a `distribution` must also be provided.  If
        `context` is provided, a ``dim_data_per_rank`` will be pulled from
        the existing ``LocalArray``\s and a ``Distribution`` will be created
        from it.   If `distribution` is provided, it should accurately
        reflect the  distribution of the existing ``LocalArray``\s.

        If `dtype` is not provided, it will be fetched from the engines.
        """

        def get_dim_datas_and_dtype(arr):
            return (arr.dim_data, arr.dtype)

        da = cls.__new__(cls)
        da.key = key

        if (context is None) == (distribution is None):
            errmsg = "Must provide `context` or `distribution` but not both."
            raise RuntimeError(errmsg)

        # has context, get dist and dtype
        elif (distribution is None) and (dtype is None):
            res = context.apply(get_dim_datas_and_dtype, args=(key,),
                                targets=targets)
            dim_datas = [i[0] for i in res]
            dtypes = [i[1] for i in res]
            da._dtype = dtypes[0]
            da.distribution = Distribution.from_dim_data_per_rank(context,
                                                                  dim_datas,
                                                                  targets)

        # has context and dtype, get dist
        elif (distribution is None) and (dtype is not None):
            da._dtype = dtype
            dim_datas = context.apply(getattr, args=(key, 'dim_data'),
                                      targets=targets)
            da.distribution = Distribution.from_dim_data_per_rank(context,
                                                                  dim_datas,
                                                                  targets)

        # has distribution, get dtype
        elif (distribution is not None) and (dtype is None):
            da.distribution = distribution
            da._dtype = distribution.context.apply(getattr,
                                                   args=(key, 'dtype'),
                                                   targets=[0])[0]
        # has distribution and dtype
        elif (distribution is not None) and (dtype is not None):
            da.distribution = distribution
            da._dtype = dtype

        # sanity check that I didn't miss any cases above, because this is a
        # confusing function
        else:
            assert False
        return da

    def __del__(self):
        try:
            self.context.delete_key(self.key, self.targets)
        except Exception:
            pass

    def __repr__(self):
        s = '<DistArray(shape=%r, targets=%r)>' % \
            (self.shape, self.targets)
        return s

    def _get_view(self, index):

        # to be run locally
        def get_view(arr, index, ddpr, comm):
            from distarray.localapi.maps import Distribution
            if len(ddpr) == 0:
                dim_data = ()
            else:
                dim_data = ddpr[comm.Get_rank()]
            local_distribution = Distribution(comm=comm, dim_data=dim_data)
            result = arr.global_index.get_slice(index, local_distribution)
            return proxyize(result)

        new_distribution = self.distribution.slice(index)
        ddpr = new_distribution.get_dim_data_per_rank()

        args = [self.key, index, ddpr, new_distribution.comm]
        targets = new_distribution.targets
        result = self.context.apply(get_view, args=args, targets=targets)[0]

        return DistArray.from_localarrays(key=result,
                                          targets=targets,
                                          distribution=new_distribution,
                                          dtype=self.dtype)

    def _get_value(self, index):

        # to be run locally
        def get_value(arr, index):
            return arr.global_index[index]

        args = [self.key, index]
        targets = self.distribution.owning_targets(index)
        result = self.context.apply(get_value, args=args, targets=targets)

        return [i for i in result if i is not None][0]

    def _checked_getitem(self, index):

        # to be run locally
        def checked_getitem(arr, index):
            return arr.global_index.checked_getitem(index)

        args = [self.key, index]
        targets = self.distribution.owning_targets(index)
        result = self.context.apply(checked_getitem, args=args, targets=targets)

        somethings = [i for i in result if i is not None]
        if len(somethings) == 0:
            # all return None
            raise IndexError("Index %r is is not present." % (index,))
        elif len(somethings) == 1:
            return somethings[0]
        else:
            return result

    def __getitem__(self, index):
        return_type, index = sanitize_indices(index, ndim=self.ndim,
                                              shape=self.shape)
        if not self.distribution.has_precise_index:
            result = self._checked_getitem(index)
        elif return_type == 'view':
            result = self._get_view(index)
        elif return_type == 'value':
            result = self._get_value(index)
        else:
            assert False

        return result

    def _set_value(self, index, value):
        # to be run locally
        def set_value(arr, index, value):
            arr.global_index[index] = value

        args = [self.key, index, value]
        targets = self.distribution.owning_targets(index)
        self.context.apply(set_value, args=args, targets=targets)

    def _set_view(self, index, value):
        # to be run locally
        def set_view(arr, index, value, ddpr, comm):
            from distarray.localapi.localarray import LocalArray
            from distarray.localapi.maps import Distribution
            if len(ddpr) == 0:
                dim_data = ()
            else:
                dim_data = ddpr[comm.Get_rank()]
            dist = Distribution(comm=comm, dim_data=dim_data)
            if isinstance(value, LocalArray):
                arr.global_index[index] = value.ndarray
            else:
                arr.global_index[index] = value[dist.global_slice]

        new_distribution = self.distribution.slice(index)
        if isinstance(value, DistArray):
            if not value.distribution.is_compatible(new_distribution):
                msg = "rvalue Distribution not compatible."
                raise ValueError(msg)
            value = value.key
        else:
            value = np.asarray(value)  # convert to array
            if value.shape != new_distribution.shape:
                msg = "Slice shape does not equal rvalue shape."
                raise ValueError(msg)

        ddpr = new_distribution.get_dim_data_per_rank()
        comm = new_distribution.comm
        targets = new_distribution.targets
        args = [self.key, index, value, ddpr, comm]
        self.context.apply(set_view, args=args, targets=targets)

    def _checked_setitem(self, index, value):
        # to be run locally
        def checked_setitem(arr, index, value):
            return arr.global_index.checked_setitem(index, value)

        args = [self.key, index, value]
        targets = self.distribution.owning_targets(index)
        result = self.context.apply(checked_setitem, args=args,
                                    targets=targets)
        result = [i for i in result if i is not None]
        if len(result) > 1:
            raise IndexError("Setting more than one result (%s) is "
                             "not supported yet." % (result,))
        elif result == []:
            raise IndexError("Index %s is out of bounds" % (index,))

    def __setitem__(self, index, value):
        set_type, index = sanitize_indices(index, ndim=self.ndim,
                                           shape=self.shape)
        if not self.distribution.has_precise_index:
            self._checked_setitem(index, value)
        elif set_type == 'view':
            self._set_view(index, value)
        elif set_type == 'value':
            self._set_value(index, value)
        else:
            assert False

    @property
    def context(self):
        return self.distribution.context

    @property
    def shape(self):
        return self.distribution.shape

    @property
    def global_size(self):
        return reduce(operator.mul, self.shape)

    @property
    def dist(self):
        return self.distribution.dist

    @property
    def grid_shape(self):
        return self.distribution.grid_shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return self.global_size * self.itemsize

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def targets(self):
        return self.distribution.targets

    @property
    def __array_interface__(self):
        return {'shape': self.shape,
                'typestr': self.dtype.str,
                'data': self.tondarray(),
                'version': 3}

    def tondarray(self):
        """Returns the distributed array as an ndarray."""
        arr = np.empty(self.shape, dtype=self.dtype)
        local_arrays = self.get_localarrays()
        try:
            for local_array in local_arrays:
                gslice = local_array.distribution.global_slice
                arr[gslice] = local_array.ndarray
        except AttributeError:
            # do it the slow way
            for local_array in local_arrays:
                maps = (list(ax_map.global_iter) for ax_map in
                        local_array.distribution)
                for index in product(*maps):
                    arr[index] = local_array.global_index[index]
        return arr

    toarray = tondarray

    def fill(self, value):
        def inner_fill(arr, value):
            arr.fill(value)
        self.context.apply(inner_fill, args=(self.key, value), targets=self.targets)

    def _reduce(self, local_reduce_name, axes=None, dtype=None, out=None):

        if any(0 in localshape for localshape in self.localshapes()):
            raise NotImplementedError("Reduction not implemented for empty "
                                      "LocalArrays")

        if out is not None:
            _raise_nie()

        dtype = dtype or self.dtype

        out_dist = self.distribution.reduce(axes=axes)
        ddpr = out_dist.get_dim_data_per_rank()

        def _local_reduce(local_name, larr, out_comm, ddpr, dtype, axes):
            import distarray.localapi.localarray as la
            local_reducer = getattr(la, local_name)
            res = proxyize(la.local_reduction(out_comm, local_reducer, larr,  # noqa
                                              ddpr, dtype, axes))
            return res

        local_reduce_args = (local_reduce_name, self.key, out_dist.comm, ddpr,
                             dtype, normalize_reduction_axes(axes, self.ndim))
        out_key = self.context.apply(_local_reduce, local_reduce_args,
                                     targets=self.targets)[0]

        return DistArray.from_localarrays(key=out_key, distribution=out_dist,
                                          dtype=dtype)

    def sum(self, axis=None, dtype=None, out=None):
        """Return the sum of array elements over the given axis."""
        if dtype is None and self.dtype == np.bool:
            dtype = np.uint64
        return self._reduce('sum_reducer', axis, dtype, out)

    def mean(self, axis=None, dtype=float, out=None):
        """Return the mean of array elements over the given axis."""
        return self._reduce('mean_reducer', axis, dtype, out)

    def var(self, axis=None, dtype=float, out=None):
        """Return the variance of array elements over the given axis."""
        return self._reduce('var_reducer', axis, dtype, out)

    def std(self, axis=None, dtype=float, out=None):
        """Return the standard deviation of array elements over the given axis."""
        return self._reduce('std_reducer', axis, dtype, out)

    def min(self, axis=None, dtype=None, out=None):
        """Return the minimum of array elements over the given axis."""
        return self._reduce('min_reducer', axis, dtype, out)

    def max(self, axis=None, dtype=None, out=None):
        """Return the maximum of array elements over the given axis."""
        return self._reduce('max_reducer', axis, dtype, out)

    def get_ndarrays(self):
        """Pull the local ndarrays from the engines.

        Returns
        -------
        list of ndarrays
            one ndarray per process

        """
        def get(key):
            return key.ndarray
        return self.context.apply(get, args=(self.key,), targets=self.targets)

    def get_localarrays(self):
        """Pull the LocalArray objects from the engines.

        Returns
        -------
        list of localarrays
            one localarray per process

        """
        def get(key):
            return key.copy()
        return self.context.apply(get, args=(self.key,), targets=self.targets)

    def localshapes(self):
        return self.distribution.localshapes()

    def view(self, dtype=None):
        """
        New view of array with the same data.

        Parameters
        ----------
        dtype : numpy dtype, optional
            Data-type descriptor of the returned view, e.g., float32 or
            int16. The default, None, results in the view having the same
            data-type as the original array.

        Returns
        -------
        DistArray
            A view on the original DistArray, optionally with the underlying
            memory interpreted as a different dtype.

        Note
        ----
        No redistribution is done.  The sizes of all `LocalArray`\s must be
        compatible with the new view.
        """
        # special case for same dtype
        if (dtype is None) or (np.dtype(dtype) == self.dtype):
            return self[...]

        def local_view(larr, ddpr, dtype):
            from distarray.localapi.maps import Distribution
            if len(ddpr) == 0:
                dim_data = ()
            else:
                dim_data = ddpr[larr.comm_rank]
            ldist = Distribution(comm=larr.comm, dim_data=dim_data)
            lview = larr.view(ldist, dtype=dtype)
            return proxyize(lview)

        old_itemsize = self.dtype.itemsize
        new_itemsize = np.dtype(dtype).itemsize
        last_dimsize = self.distribution[-1].size
        errmsg = "New dtype not compatible with existing DistArray dtype."
        if old_itemsize == new_itemsize:
            # no scaling
            new_dimsize = last_dimsize
        elif old_itemsize % new_itemsize == 0:
            # easy case: scale a dimension up
            new_dimsize = (old_itemsize * last_dimsize) / new_itemsize
        elif (old_itemsize * last_dimsize) % new_itemsize == 0:
            # harder case: scale a dimension if local array shapes allow it
            # check local last-dimension size compatibility:
            local_lastdim_sizes = np.array([s[-1] for s in self.localshapes()])
            compat =  (old_itemsize * local_lastdim_sizes) % new_itemsize == 0
            if np.all(compat):
                new_dimsize = (old_itemsize * last_dimsize) / new_itemsize
            else:
                raise ValueError(errmsg)
        else:
            raise ValueError(errmsg)

        new_dist = self.distribution.view(new_dimsize)
        ddpr = new_dist.get_dim_data_per_rank()
        new_key = self.context.apply(local_view, (self.key, ddpr, dtype))[0]
        return DistArray.from_localarrays(key=new_key, distribution=new_dist,
                                          dtype=dtype)

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
        return self._binary_op_from_ufunc(other, distarray.globalapi.add, '__radd__', *args, **kwargs)

    def __sub__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.subtract, '__rsub__', *args, **kwargs)

    def __mul__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.multiply, '__rmul__', *args, **kwargs)

    def __div__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.divide, '__rdiv__', *args, **kwargs)

    def __truediv__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.true_divide, '__rtruediv__', *args, **kwargs)

    def __floordiv__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.floor_divide, '__rfloordiv__', *args, **kwargs)

    def __mod__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.mod, '__rdiv__', *args, **kwargs)

    def __pow__(self, other, modulo=None, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.power, '__rpower__', *args, **kwargs)

    def __lshift__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.left_shift, '__rlshift__', *args, **kwargs)

    def __rshift__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.right_shift, '__rrshift__', *args, **kwargs)

    def __and__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.bitwise_and, '__rand__', *args, **kwargs)

    def __or__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.bitwise_or, '__ror__', *args, **kwargs)

    def __xor__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.bitwise_xor, '__rxor__', *args, **kwargs)

    # Binary - right versions

    def __radd__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.add, '__add__', *args, **kwargs)

    def __rsub__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.subtract, '__sub__', *args, **kwargs)

    def __rmul__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.multiply, '__mul__', *args, **kwargs)

    def __rdiv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.divide, '__div__', *args, **kwargs)

    def __rtruediv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.true_divide, '__truediv__', *args, **kwargs)

    def __rfloordiv__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.floor_divide, '__floordiv__', *args, **kwargs)

    def __rmod__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.mod, '__mod__', *args, **kwargs)

    def __rpow__(self, other, modulo=None, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.power, '__pow__', *args, **kwargs)

    def __rlshift__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.left_shift, '__lshift__', *args, **kwargs)

    def __rrshift__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.right_shift, '__rshift__', *args, **kwargs)

    def __rand__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.bitwise_and, '__and__', *args, **kwargs)

    def __ror__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.bitwise_or, '__or__', *args, **kwargs)

    def __rxor__(self, other, *args, **kwargs):
        return self._rbinary_op_from_ufunc(other, distarray.globalapi.bitwise_xor, '__xor__', *args, **kwargs)

    def __neg__(self, *args, **kwargs):
        return distarray.globalapi.negative(self, *args, **kwargs)

    def __pos__(self, *args, **kwargs):
        return self

    def __abs__(self, *args, **kwargs):
        return distarray.globalapi.absolute(self, *args, **kwargs)

    def __invert__(self, *args, **kwargs):
        return distarray.globalapi.invert(self, *args, **kwargs)

    # Boolean comparisons

    def __lt__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.less, '__lt__', *args, **kwargs)

    def __le__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.less_equal, '__le__', *args, **kwargs)

    def __eq__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.equal, '__eq__', *args, **kwargs)

    def __ne__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.not_equal, '__ne__', *args, **kwargs)

    def __gt__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.greater, '__gt__', *args, **kwargs)

    def __ge__(self, other, *args, **kwargs):
        return self._binary_op_from_ufunc(other, distarray.globalapi.greater_equal, '__ge__', *args, **kwargs)
