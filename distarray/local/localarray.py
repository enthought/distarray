# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from __future__ import print_function, division

from distarray import metadata_utils

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------
import math
import operator
from functools import reduce
from collections import Mapping

import numpy as np

from distarray.externals import six
from distarray.externals.six import next
from distarray.externals.six.moves import zip, range

from distarray.mpiutils import MPI
from distarray.utils import _raise_nie
from distarray.local import construct, format, maps
from distarray.local.error import InvalidDimensionError, IncompatibleArrayError

def _start_stop_block(size, proc_grid_size, proc_grid_rank):
    nelements = size // proc_grid_size
    if size % proc_grid_size != 0:
        nelements += 1

    start = proc_grid_rank * nelements
    if start > size:
        start = size
        stop = size

    stop = start + nelements
    if stop > size:
        stop = size

    return start, stop

def distribute_indices(dim_data):
    """Fill in missing index related keys...

    for supported dist_types.
    """
    distribute_fn = {
        'n': lambda dd: None,
        'b': distribute_block_indices,
        'c': distribute_cyclic_indices,
        'u': lambda dd: None,
        }
    for dim in dim_data:
        distribute_fn[dim['dist_type']](dim)


def distribute_cyclic_indices(dd):
    """Fill in `start` given dimdict `dd`."""
    if 'start' in dd:
        return
    else:
        dd['start'] = dd['proc_grid_rank']


def distribute_block_indices(dd):
    """Fill in `start` and `stop` in dimdict `dd`."""
    if ('start' in dd) and ('stop' in dd):
        return

    nelements = dd['size'] // dd['proc_grid_size']
    if dd['size'] % dd['proc_grid_size'] != 0:
        nelements += 1

    dd['start'] = dd['proc_grid_rank'] * nelements
    if dd['start'] > dd['size']:
        dd['start'] = dd['size']
        dd['stop'] = dd['size']

    dd['stop'] = dd['start'] + nelements
    if dd['stop'] > dd['size']:
        dd['stop'] = dd['size']

def _normalize_dim_data(dim_data):
    ''' Adds `proc_grid_size` and `proc_grid_rank` for 'n' disttype.'''
    for dd in dim_data:
        if dd['dist_type'] == 'n':
            dd['proc_grid_size'] = 1
            dd['proc_grid_rank'] = 0
    return dim_data


def make_partial_dim_data(shape, dist=None, grid_shape=None):
    """Create an (incomplete) dim_data structure from simple parameters.

    Parameters
    ----------
    shape : tuple of int
        Number of elements in each dimension.
    dist : dict mapping int -> str, default is {0: 'b'}
        Keys are dimension number, values are dist_type, e.g 'b', 'c', or 'n'.
    grid_shape : tuple of int, optional
        Size of process grid in each dimension

    Returns
    -------
    dim_data : tuple of dict
        Partial dim_data structure as outlined in the Distributed Array
        Protocol.
    """
    supported_dist_types = ('n', 'b', 'c')

    if dist is None:
        dist = {0: 'b'}

    dist_tuple = metadata_utils.normalize_dist(dist, len(shape))

    if grid_shape:  # if None, LocalArray will initialize
        grid_gen = iter(grid_shape)

    dim_data = []
    for size, dist_type in zip(shape, dist_tuple):
        if dist_type not in supported_dist_types:
            msg = "dist_type {} not supported. Try `from_dim_data`."
            raise TypeError(msg.format(dist_type))
        dimdict = dict(dist_type=dist_type, size=size)
        if grid_shape is not None and dist_type != 'n':
            dimdict["proc_grid_size"] = next(grid_gen)

        dim_data.append(dimdict)

    return tuple(dim_data)


class LocalArray(object):

    """Distributed memory Python arrays."""

    __array_priority__ = 20.0

    #-------------------------------------------------------------------------
    # Methods used for initialization
    #-------------------------------------------------------------------------

    def _init(self, dim_data, dtype=None, buf=None, comm=None):
        """Private init method."""
        self.dim_data = _normalize_dim_data(dim_data)
        self.base_comm = construct.init_base_comm(comm)

        self._init_grid_shape()

        self.comm = construct.init_comm(self.base_comm, self.grid_shape)

        self._cache_proc_grid_rank()
        distribute_indices(self.dim_data)
        self.maps = tuple(maps.IndexMap.from_dimdict(dimdict)
                          for dimdict in dim_data)

        self.local_array = self._make_local_array(buf=buf, dtype=dtype)

        self.base = None
        self.ctypes = None

    def _init_grid_shape(self):

        grid_shape = metadata_utils.make_grid_shape(self.global_shape,
                                                    self.dist,
                                                    self.comm_size)
        metadata_utils.validate_grid_shape(grid_shape,
                                           self.dist,
                                           self.comm_size)

        for gs, dd in zip(grid_shape, self.dim_data):
            dd['proc_grid_size'] = gs


    @classmethod
    def from_dim_data(cls, dim_data, dtype=None, buf=None, comm=None):
        """Make a LocalArray from a `dim_data` tuple.

        Parameters
        ----------
        dim_data : tuple of dictionaries
            A dict for each dimension, with the data described here:
            https://github.com/enthought/distributed-array-protocol
        dtype : numpy dtype, optional
            If both `dtype` and `buf` are provided, `buf` will be
            encapsulated and interpreted with the given dtype.  If neither
            are, an empty array will be created with a dtype of 'float'.  If
            only `dtype` is given, an empty array of that dtype will be
            created.
        buf : buffer object, optional
            If both `dtype` and `buf` are provided, `buf` will be
            encapsulated and interpreted with the given dtype.  If neither
            are, an empty array will be created with a dtype of 'float'.  If
            only `buf` is given, `self.dtype` will be set to its dtype.
        comm : MPI comm object, optional

        Returns
        -------
        LocalArray
            A LocalArray encapsulating `buf`, or else an empty
            (uninitialized) LocalArray.
        """
        self = cls.__new__(cls)
        self._init(dim_data=dim_data, dtype=dtype, buf=buf, comm=comm)
        return self

    def __init__(self, shape, dtype=None, dist=None, grid_shape=None,
                 comm=None, buf=None):
        """Create a LocalArray from a simple set of parameters.

        This initializer restricts you to 'b' and 'c' dist_types and evenly
        distributed data.  See `LocalArray.from_dim_data` for a more general
        method.

        Parameters
        ----------
        shape : tuple of int
            Number of elements in each dimension.
        dtype : numpy dtype, optional
        dist : dict mapping int -> str, default is {0: 'b'}, optional
            Keys are dimension number, values are dist_type, e.g 'b', 'c', or
            'n'.
        grid_shape : tuple of int, optional
            A size of each dimension of the process grid.
            There should be a dimension size for each distributed
            dimension in `dist`.
        comm : MPI communicator object, optional
        buf : buffer object, optional
            If not given, an empty array is created.

        See also
        --------
        LocalArray.from_dim_data
        """
        dim_data = make_partial_dim_data(shape=shape, dist=dist,
                                         grid_shape=grid_shape)
        self._init(dim_data=dim_data, dtype=dtype, buf=buf, comm=comm)

    def __del__(self):
        # If the __init__ method fails, we may not have a valid comm
        # attribute and this needs to be protected against.
        if hasattr(self, 'comm'):
            if self.comm is not None:
                try:
                    self.comm.Free()
                except:
                    pass

    @property
    def local_shape(self):
        return tuple(m.size for m in self.maps)

    @property
    def grid_shape(self):
        return tuple(dd['proc_grid_size'] for dd in self.dim_data)

    @property
    def global_shape(self):
        return tuple(dd['size'] for dd in self.dim_data)

    @property
    def ndim(self):
        return len(self.dim_data)

    @property
    def global_size(self):
        return reduce(operator.mul, self.global_shape)

    @property
    def comm_size(self):
        return self.base_comm.Get_size()

    @property
    def comm_rank(self):
        return self.base_comm.Get_rank()

    @property
    def dist(self):
        return tuple(dd['dist_type'] for dd in self.dim_data)

    @property
    def cart_coords(self):
        rval = tuple(dd['proc_grid_rank'] for dd in self.dim_data)
        assert rval == tuple(self.comm.Get_coords(self.comm_rank))
        return rval

    @property
    def local_size(self):
        return self.local_array.size

    @property
    def local_data(self):
        return self.local_array.data

    @property
    def dtype(self):
        return self.local_array.dtype

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.global_size * self.itemsize

    def _cache_proc_grid_rank(self):
        cart_coords = self.comm.Get_coords(self.comm_rank)
        assert len(cart_coords) == len(self.dim_data)
        for dim, cart_rank in zip(self.dim_data, cart_coords):
            dim['proc_grid_rank'] = cart_rank

    def _make_local_array(self, buf=None, dtype=None):
        """Encapsulate `buf` or create an empty local array.

        Returns
        -------
        local_array : numpy array
        """
        if buf is None:
            return np.empty(self.local_shape, dtype=dtype)
        else:
            mv = memoryview(buf)
            return np.asarray(mv, dtype=dtype)

    def compatibility_hash(self):
        return hash((self.global_shape, self.dist, self.grid_shape, True))
    #-------------------------------------------------------------------------
    # Distributed Array Protocol
    #-------------------------------------------------------------------------

    @classmethod
    def from_distarray(cls, obj, comm=None):
        """Make a LocalArray from Distributed Array Protocol data structure.

        An object that supports the Distributed Array Protocol will have
        a `__distarray__` method that returns the data structure
        described here:

        https://github.com/enthought/distributed-array-protocol

        Parameters
        ----------
        obj : an object with a `__distarray__` method or a dict
            If a dict, it must conform to the structure defined by the
            distributed array protocol.

        Returns
        -------
        LocalArray
            A LocalArray encapsulating the buffer of the original data.
            No copy is made.
        """
        if isinstance(obj, Mapping):
            distbuffer = obj
        else:
            distbuffer = obj.__distarray__()
        buf = np.asarray(distbuffer['buffer'])
        dim_data = distbuffer['dim_data']

        return cls.from_dim_data(dim_data=dim_data, buf=buf, comm=comm)

    def __distarray__(self):
        """Returns the data structure required by the DAP.

        DAP = Distributed Array Protocol

        See the project's documentation for the Protocol's specification.
        """
        distbuffer = {
            "__version__": "1.0.0",
            "buffer": self.local_array,
            "dim_data": self.dim_data,
            }
        return distbuffer

    #-------------------------------------------------------------------------
    # Methods related to distributed indexing
    #-------------------------------------------------------------------------

    def get_localarray(self):
        return self.local_view()

    def set_localarray(self, a):
        arr = np.asarray(a, dtype=self.dtype, order='C')
        if arr.shape == self.local_shape:
            self.local_array = arr
        else:
            raise ValueError("Incompatible local array shape")

    def coords_from_rank(self, rank):
        return self.comm.Get_coords(rank)

    def rank_from_coords(self, coords):
        return self.comm.Get_cart_rank(coords)

    def local_from_global(self, *global_ind):
        return tuple(self.maps[dim].local_index[global_ind[dim]]
                     for dim in range(self.ndim))

    def global_from_local(self, *local_ind):
        return tuple(self.maps[dim].global_index[local_ind[dim]]
                     for dim in range(self.ndim))

    def global_limits(self, dim):
        if dim < 0 or dim >= self.ndim:
            raise InvalidDimensionError("Invalid dimension: %r" % dim)
        lower_local = self.ndim * [0]
        lower_global = self.global_from_local(*lower_local)
        upper_local = [shape-1 for shape in self.local_shape]
        upper_global = self.global_from_local(*upper_local)
        return lower_global[dim], upper_global[dim]

    #-------------------------------------------------------------------------
    # 3.2 ndarray methods
    #-------------------------------------------------------------------------
    # 3.2.1 Array conversion
    #-------------------------------------------------------------------------

    def astype(self, newdtype):
        """Return a copy of this LocalArray with a new underlying dtype."""
        if newdtype is None:
            return self.copy()
        else:
            local_copy = self.local_array.astype(newdtype)
            new_da = self.__class__.from_dim_data(dim_data=self.dim_data,
                                                  dtype=newdtype,
                                                  comm=self.base_comm,
                                                  buf=local_copy)
            return new_da

    def copy(self):
        """Return a copy of this LocalArray."""
        local_copy = self.local_array.copy()
        return self.__class__.from_dim_data(dim_data=self.dim_data,
                                            dtype=self.dtype,
                                            comm=self.base_comm,
                                            buf=local_copy)

    def local_view(self, dtype=None):
        if dtype is None:
            return self.local_array.view()
        else:
            return self.local_array.view(dtype)

    def view(self, dtype=None):
        """Return a new LocalArray whose underlying `local_array` is a view on
        `self.local_array`.

        Note
        ----
        Currently unimplemented for ``dtype is not None``.
        """
        if dtype is None:
            new_da = self.__class__.from_dim_data(dim_data=self.dim_data,
                                                  dtype=self.dtype,
                                                  comm=self.base_comm,
                                                  buf=self.local_array)
        else:
            _raise_nie()
            #TODO: to implement this properly, a new dim_data will need to
            #      generated that reflects the size and shape of the new dtype.
            #new_da = self.__class__.from_dim_data(dim_data=self.dim_data,
            #                                      dtype=dtype,
            #                                      comm=self.base_comm,
            #                                      buf=self.local_array)
        return new_da

    def __array__(self, dtype=None):
        if dtype is None:
            return self.local_array
        elif np.dtype(dtype) == self.dtype:
            return self.local_array
        else:
            return self.local_array.astype(dtype)

    def __array_wrap__(self, obj, context=None):
        """
        Return a LocalArray based on obj.

        This method constructs a new LocalArray object using (shape, dist,
        grid_shape and base_comm) from self and dtype, buffer from obj.

        This is used to construct return arrays for ufuncs.
        """
        return self.__class__(self.global_shape, obj.dtype, self.dist,
                              self.grid_shape, self.base_comm, buf=obj)

    def fill(self, scalar):
        self.local_array.fill(scalar)

    #-------------------------------------------------------------------------
    # 3.2.2 Array shape manipulation
    #-------------------------------------------------------------------------

    def reshape(self, newshape):
        _raise_nie()

    def redist(self, newshape, newdist={0: 'b'}, newgrid_shape=None):
        _raise_nie()

    def resize(self, newshape, refcheck=1, order='C'):
        _raise_nie()

    def transpose(self, arg):
        _raise_nie()

    def swapaxes(self, axis1, axis2):
        _raise_nie()

    def flatten(self, order='C'):
        _raise_nie()

    def ravel(self, order='C'):
        _raise_nie()

    def squeeze(self):
        _raise_nie()

    def asdist(self, shape, dist={0: 'b'}, grid_shape=None):
        pass
        # new_da = LocalArray(shape, self.dtype, dist, grid_shape,
        #                     self.base_comm)
        # base_comm = self.base_comm
        # local_array = self.local_array
        # new_local_array = da.local_array
        # recv_counts = np.zeros(self.comm_size, dtype=int)
        #
        # status = MPI.Status()
        # MPI.Attach_buffer(np.empty(128+MPI.BSEND_OVERHEAD,dtype=float))
        # done_count = 0
        #
        # for old_local_inds, item in np.ndenumerate(local_array):
        #
        #     # Compute the new owner
        #     global_inds = self.global_from_local(new_da.comm_rank,
        #                                        old_local_inds)
        #     new_owner = new_da.owner_rank(global_inds)
        #     if new_owner==self.owner_rank:
        #         pass
        #         # Just move the data to the right place in new_local_array
        #     else:
        #         # Send to the new owner with default tag
        #         # Bsend is probably best, but Isend is also a possibility.
        #         request = comm.Isend(item, dest=new_owner)
        #
        #     # Recv
        #     incoming = comm.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
        #     if incoming:
        #         old_owner = status.Get_source()
        #         tag = status.Get_tag()
        #         data = comm.Recv(old_owner, tag)
        #         if tag==2:
        #             done_count += 1
        #         # Figure out where new location of old_owner, tag
        #         new_local_ind = local_ind_by_owner_and_location(old_owner,
        #                                                         location)
        #         new_local_array[new_local_ind] = y
        #         recv_counts[old_owner] = recv_counts[old_owner]+1
        #
        # while done_count < self.comm_size:
        #     pass
        #
        #
        # MPI.Detach_buffer()

    def asdist_like(self, other):
        """
        Return a version of self that has shape, dist and grid_shape like
        other.
        """
        if arecompatible(self, other):
            return self
        else:
            raise IncompatibleArrayError("DistArrays have incompatible shape,"
                                         "dist or grid_shape")

    #-------------------------------------------------------------------------
    # 3.2.3 Array item selection and manipulation
    #-------------------------------------------------------------------------

    def take(self, indices, axis=None, out=None, mode='raise'):
        _raise_nie()

    def put(self, values, indices, mode='raise'):
        _raise_nie()

    def putmask(self, values, mask):
        _raise_nie()

    def repeat(self, repeats, axis=None):
        _raise_nie()

    def choose(self, choices, out=None, mode='raise'):
        _raise_nie()

    def sort(self, axis=-1, kind='quick'):
        _raise_nie()

    def argsort(self, axis=-1, kind='quick'):
        _raise_nie()

    def searchsorted(self, values):
        _raise_nie()

    def nonzero(self):
        _raise_nie()

    def compress(self, condition, axis=None, out=None):
        _raise_nie()

    def diagonal(self, offset=0, axis1=0, axis2=1):
        _raise_nie()

    #-------------------------------------------------------------------------
    # 3.2.4 Array item selection and manipulation
    #-------------------------------------------------------------------------

    def max(self, axis=None, out=None):
        _raise_nie()

    def argmax(self, axis=None, out=None):
        _raise_nie()

    def min(axis=None, out=None):
        _raise_nie()

    def argmin(self, axis=None, out=None):
        _raise_nie()

    def ptp(self, axis=None, out=None):
        _raise_nie()

    def clip(self, min, max, out=None):
        _raise_nie()

    def conj(self, out=None):
        _raise_nie()

    conjugate = conj

    def round(self, decimals=0, out=None):
        _raise_nie()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        _raise_nie()

    #TODO FIXME: implement axis and out kwargs.
    def sum(self, axis=None, dtype=None, out=None):
        if axis or out is not None:
            _raise_nie()
        return sum(self, dtype=dtype)

    def mean(self, axis=None, dtype=float, out=None):
        if axis or out is not None:
            _raise_nie()
        elif dtype is not None:
            dtype = np.dtype(dtype)
            return dtype.type((np.divide(self.sum(dtype=dtype), self.global_size)))
        else:
            return np.divide(self.sum(dtype=dtype), self.global_size)

    def var(self, axis=None, dtype=None, out=None):
        if axis or out is not None:
            _raise_nie()
        mu = self.mean()
        temp = (self - mu)**2
        return temp.mean(dtype=dtype)

    def std(self, axis=None, dtype=None, out=None):
        if axis or out is not None:
            _raise_nie()
        elif dtype is not None:
            dtype = np.dtype(dtype)
            return dtype.type((math.sqrt(self.var())))
        else:
            return math.sqrt(self.var())

    def cumsum(self, axis=None, dtype=None, out=None):
        _raise_nie()

    def prod(self, axis=None, dtype=None, out=None):
        _raise_nie()

    def cumprod(self, axis=None, dtype=None, out=None):
        _raise_nie()

    def all(self, axis=None, out=None):
        _raise_nie()

    def any(self, axis=None, out=None):
        _raise_nie()

    #-------------------------------------------------------------------------
    # 3.3 Array special methods
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # 3.3.1 Methods for standard library functions
    #-------------------------------------------------------------------------

    def __copy__(self):
        _raise_nie()

    def __deepcopy__(self):
        _raise_nie()

    #-------------------------------------------------------------------------
    # 3.3.2 Basic customization
    #-------------------------------------------------------------------------

    def __lt__(self, other):
        return self._binary_op_from_ufunc(other, less, '__lt__')

    def __le__(self, other):
        return self._binary_op_from_ufunc(other, less_equal, '__le__')

    def __eq__(self, other):
        return self._binary_op_from_ufunc(other, equal, '__eq__')

    def __ne__(self, other):
        return self._binary_op_from_ufunc(other, not_equal, '__ne__')

    def __gt__(self, other):
        return self._binary_op_from_ufunc(other, greater, '__gt__')

    def __ge__(self, other):
        return self._binary_op_from_ufunc(other, greater_equal, '__ge__')

    def __str__(self):
        return str(self.local_array)

    def __repr__(self):
        return str(self.local_array)

    def __nonzero__(self):
        _raise_nie()

    #-------------------------------------------------------------------------
    # 3.3.3 Container customization
    #-------------------------------------------------------------------------

    def __len__(self):
        return self.global_shape[0]

    def checked_getitem(self, global_inds):
        try:
            return self.__getitem__(global_inds)
        except IndexError:
            return None

    def checked_setitem(self, global_inds, value):
        try:
            self.__setitem__(global_inds, value)
            return True
        except IndexError:
            return None

    def _sanitize_indices(self, indices):
        if isinstance(indices, int) or isinstance(indices, slice):
            return (indices,)
        elif all(isinstance(i, int) or isinstance(i, slice) for i in indices):
            return indices
        else:
            raise TypeError("Index must be a sequence of ints and slices")

    def __getitem__(self, global_inds):
        global_inds = self._sanitize_indices(global_inds)
        try:
            local_inds = self.local_from_global(*global_inds)
            return self.local_array[local_inds]
        except KeyError as err:
            raise IndexError(err)

    def __setitem__(self, global_inds, value):
        global_inds = self._sanitize_indices(global_inds)
        try:
            local_inds = self.local_from_global(*global_inds)
            self.local_array[local_inds] = value
        except KeyError as err:
            raise IndexError(err)

    def sync(self):
        raise NotImplementedError("`sync` not yet implemented.")

    def __contains__(self, item):
        return item in self.local_array

    def pack_index(self, inds):
        inds_array = np.array(inds)
        strides_array = np.cumprod([1] + list(self.global_shape)[:0:-1])[::-1]
        return np.sum(inds_array*strides_array)

    def unpack_index(self, packed_ind):
        if packed_ind > np.prod(self.global_shape)-1 or packed_ind < 0:
            raise ValueError("Invalid index, must be 0 <= x <= number of"
                             "elements.")
        strides_array = np.cumprod([1] + list(self.global_shape)[:0:-1])[::-1]
        return tuple(packed_ind//strides_array % self.global_shape)

    #--------------------------------------------------------------------------
    # 3.3.4 Arithmetic customization - binary
    #--------------------------------------------------------------------------

    # Binary

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
        return self._binary_op_from_ufunc(other, add, '__radd__')

    def __sub__(self, other):
        return self._binary_op_from_ufunc(other, subtract, '__rsub__')

    def __mul__(self, other):
        return self._binary_op_from_ufunc(other, multiply, '__rmul__')

    def __div__(self, other):
        return self._binary_op_from_ufunc(other, divide, '__rdiv__')

    def __truediv__(self, other):
        return self._binary_op_from_ufunc(other, true_divide, '__rtruediv__')

    def __floordiv__(self, other):
        return self._binary_op_from_ufunc(other, floor_divide, '__rfloordiv__')

    def __mod__(self, other):
        return self._binary_op_from_ufunc(other, mod, '__rdiv__')

    def __divmod__(self, other):
        _raise_nie()

    def __pow__(self, other, modulo=None):
        return self._binary_op_from_ufunc(other, power, '__rpower__')

    def __lshift__(self, other):
        return self._binary_op_from_ufunc(other, left_shift, '__rlshift__')

    def __rshift__(self, other):
        return self._binary_op_from_ufunc(other, right_shift, '__rrshift__')

    def __and__(self, other):
        return self._binary_op_from_ufunc(other, bitwise_and, '__rand__')

    def __or__(self, other):
        return self._binary_op_from_ufunc(other, bitwise_or, '__ror__')

    def __xor__(self, other):
        return self._binary_op_from_ufunc(other, bitwise_xor, '__rxor__')

    # Binary - right versions

    def __radd__(self, other):
        return self._rbinary_op_from_ufunc(other, add, '__add__')

    def __rsub__(self, other):
        return self._rbinary_op_from_ufunc(other, subtract, '__sub__')

    def __rmul__(self, other):
        return self._rbinary_op_from_ufunc(other, multiply, '__mul__')

    def __rdiv__(self, other):
        return self._rbinary_op_from_ufunc(other, divide, '__div__')

    def __rtruediv__(self, other):
        return self._rbinary_op_from_ufunc(other, true_divide, '__truediv__')

    def __rfloordiv__(self, other):
        return self._rbinary_op_from_ufunc(other, floor_divide, '__floordiv__')

    def __rmod__(self, other):
        return self._rbinary_op_from_ufunc(other, mod, '__mod__')

    def __rdivmod__(self, other):
        _raise_nie()

    def __rpow__(self, other, modulo=None):
        return self._rbinary_op_from_ufunc(other, power, '__pow__')

    def __rlshift__(self, other):
        return self._rbinary_op_from_ufunc(other, left_shift, '__lshift__')

    def __rrshift__(self, other):
        return self._rbinary_op_from_ufunc(other, right_shift, '__rshift__')

    def __rand__(self, other):
        return self._rbinary_op_from_ufunc(other, bitwise_and, '__and__')

    def __ror__(self, other):
        return self._rbinary_op_from_ufunc(other, bitwise_or, '__or__')

    def __rxor__(self, other):
        return self._rbinary_op_from_ufunc(other, bitwise_xor, '__xor__')

    # Inplace

    def __iadd__(self, other):
        _raise_nie()

    def __isub__(self, other):
        _raise_nie()

    def __imul__(self, other):
        _raise_nie()

    def __idiv__(self, other):
        _raise_nie()

    def __itruediv__(self, other):
        _raise_nie()

    def __ifloordiv__(self, other):
        _raise_nie()

    def __imod__(self, other):
        _raise_nie()

    def __ipow__(self, other, modulo=None):
        _raise_nie()

    def __ilshift__(self, other):
        _raise_nie()

    def __irshift__(self, other):
        _raise_nie()

    def __iand__(self, other):
        _raise_nie()

    def __ior__(self, other):
        _raise_nie()

    def __ixor__(self, other):
        _raise_nie()

    # Unary

    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return abs(self)

    def __invert__(self):
        return invert(self)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Functions that are friends of LocalArray
#
# I would really like these functions to be in a separate file, but that
# is not possible because of circular import problems.  Basically, these
# functions need access to the LocalArray object in this module, and the
# LocalArray object needs to use these functions.  There are 3 options for
# solving this problem:
#
#     * Put everything in one file
#     * Put the functions needed by LocalArray in distarray, others elsewhere
#     * Make a subclass of LocalArray that has methods that use the functions
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Utilities needed to implement things below
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 4 Basic routines
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 4.1 Creating arrays
#----------------------------------------------------------------------------

def aslocalarray(object, dtype=None, order=None):
    _raise_nie()


def arange(start, stop=None, step=1, dtype=None, dist={0: 'b'},
           grid_shape=None, comm=None):
    _raise_nie()


def empty(shape, dtype=float, dist=None, grid_shape=None, comm=None):
    return LocalArray(shape, dtype=dtype, dist=dist, grid_shape=grid_shape,
                      comm=comm)


def empty_like(arr, dtype=None):
    if isinstance(arr, LocalArray):
        if dtype is None:
            return empty(arr.global_shape, arr.dtype, arr.dist, arr.grid_shape,
                         arr.base_comm)
        else:
            return empty(arr.global_shape, dtype, arr.dist, arr.grid_shape,
                         arr.base_comm)
    else:
        raise TypeError("A LocalArray or subclass is expected")


def zeros(shape, dtype=float, dist=None, grid_shape=None, comm=None):
    la = LocalArray(shape, dtype, dist, grid_shape, comm)
    la.fill(0)
    return la


def zeros_like(arr):
    if isinstance(arr, LocalArray):
        return zeros(arr.global_shape, arr.dtype, arr.dist, arr.grid_shape,
                     arr.base_comm)
    else:
        raise TypeError("A LocalArray or subclass is expected")


def ones(shape, dtype=float, dist=None, grid_shape=None, comm=None):
    la = LocalArray(shape, dtype, dist, grid_shape, comm)
    la.fill(1)
    return la


def save_dnpy(file, arr):
    """
    Save a LocalArray to a ``.dnpy`` file.

    Parameters
    ----------
    file : file-like object or str
        The file or filename to which the data is to be saved.
    arr : LocalArray
        Array to save to a file.

    """
    own_fid = False
    if isinstance(file, six.string_types):
        fid = open(file, "wb")
        own_fid = True
    else:
        fid = file

    try:
        format.write_localarray(fid, arr)
    finally:
        if own_fid:
            fid.close()


def load_dnpy(file, comm=None):
    """
    Load a LocalArray from a ``.dnpy`` file.

    Parameters
    ----------
    file : file-like object or str
        The file to read.  It must support ``seek()`` and ``read()`` methods.

    Returns
    -------
    result : LocalArray
        A LocalArray encapsulating the data loaded.

    """
    own_fid = False
    if isinstance(file, six.string_types):
        fid = open(file, "rb")
        own_fid = True
    else:
        fid = file

    try:
        distbuffer = format.read_localarray(fid)
        return LocalArray.from_distarray(distbuffer, comm=comm)

    finally:
        if own_fid:
            fid.close()


def save_hdf5(filename, arr, key='buffer', mode='a'):
    """
    Save a LocalArray to a dataset in an ``.hdf5`` file.

    Parameters
    ----------
    filename : str
        Name of file to write to.
    arr : LocalArray
        Array to save to a file.
    key : str, optional
        The identifier for the group to save the LocalArray to (the default is
        'buffer').
    mode : optional, {'w', 'w-', 'a'}, default 'a'

        ``'w'``
            Create file, truncate if exists
        ``'w-'``
            Create file, fail if exists
        ``'a'``
            Read/write if exists, create otherwise (default)

    """
    try:
        import h5py
    except ImportError:
        errmsg = "An MPI-enabled h5py must be available to use save_hdf5."
        raise ImportError(errmsg)

    with h5py.File(filename, mode, driver='mpio', comm=arr.comm) as fp:
        dset = fp.create_dataset(key, arr.global_shape, dtype=arr.dtype)
        for index, value in ndenumerate(arr):
            dset[index] = value


def compact_indices(dim_data):
    """Given a `dim_data` structure, return a tuple of compact indices.

    For every dimension in `dim_data`, return a representation of the indicies
    indicated by that dim_dict; return a slice if possible, else, return the
    list of global indices.

    Parameters
    ----------
    dim_data : tuple of dict
        A dict for each dimension, with the data described here:
        https://github.com/enthought/distributed-array-protocol we use only the
        indexing related keys from this structure here.

    Returns
    -------
    index : tuple of slices and/or lists of int
        Efficient structure usable for indexing into a numpy-array-like data
        structure.

    """
    def nodist_index(dd):
        return slice(None)

    def block_index(dd):
        return slice(dd['start'], dd['stop'])

    def cyclic_index(dd):
        if ('block_size' not in dd) or (dd['block_size'] == 1):
            return slice(dd['start'], None, dd['proc_grid_size'])
        else:
            return maps.IndexMap.from_dimdict(dd).global_index

    def unstructured_index(dd):
        return maps.IndexMap.from_dimdict(dd).global_index

    index_fn_map = {'n': nodist_index,
                    'b': block_index,
                    'c': cyclic_index,
                    'u': unstructured_index,
                    }

    index = []
    for dd in dim_data:
        index_fn = index_fn_map[dd['dist_type']]
        index.append(index_fn(dd))

    return tuple(index)


def load_hdf5(filename, dim_data, key='buffer', comm=None):
    """
    Load a LocalArray from an ``.hdf5`` file.

    Parameters
    ----------
    filename : str
        The filename to read.
    dim_data : tuple of dict
        A dict for each dimension, with the data described here:
        https://github.com/enthought/distributed-array-protocol, describing
        which portions of the HDF5 file to load into this LocalArray, and with
        what metadata.
    key : str, optional
        The identifier for the group to load the LocalArray from (the default
        is 'buffer').
    comm : MPI comm object, optional

    Returns
    -------
    result : LocalArray
        A LocalArray encapsulating the data loaded.

    Note
    ----
    For `dim_data` dimension dictionaries containing unstructured ('u')
    distribution types, the indices selected by the `'indices'` key must be in
    increasing order.  This is a limitation of h5py / hdf5.

    """
    try:
        import h5py
    except ImportError:
        errmsg = "An MPI-enabled h5py must be available to use save_hdf5."
        raise ImportError(errmsg)

    #TODO: validate dim_data somehow
    index = compact_indices(dim_data)

    with h5py.File(filename, mode='r', driver='mpio', comm=comm) as fp:
        dset = fp[key]
        buf = dset[index]
        dtype = dset.dtype

    return LocalArray.from_dim_data(dim_data, dtype=dtype, buf=buf, comm=comm)


def load_npy(filename, dim_data, comm=None):
    """
    Load a LocalArray from a ``.npy`` file.

    Parameters
    ----------
    filename : str
        The file to read.
    dim_data : tuple of dict
        A dict for each dimension, with the data described here:
        https://github.com/enthought/distributed-array-protocol, describing
        which portions of the HDF5 file to load into this LocalArray, and with
        what metadata.
    comm : MPI comm object, optional

    Returns
    -------
    result : LocalArray
        A LocalArray encapsulating the data loaded.

    """
    #TODO: validate dim_data somehow
    index = compact_indices(dim_data)
    data = np.load(filename, mmap_mode='r')
    buf = data[index].copy()

    # Apparently there isn't a clean way to close a numpy memmap; it is closed
    # when the object is garbage-collected.  This stackoverflow question claims
    # that one can close it with data._mmap.close(), but it seems risky
    # http://stackoverflow.com/questions/6397495/unmap-of-numpy-memmap

    #data._mmap.close()
    return LocalArray.from_dim_data(dim_data, dtype=data.dtype, buf=buf,
                                    comm=comm)


class GlobalIterator(six.Iterator):

    def __init__(self, arr):
        self.arr = arr
        self.nditerator = np.ndenumerate(self.arr.local_view())

    def __iter__(self):
        return self

    def __next__(self):
        local_inds, value = six.advance_iterator(self.nditerator)
        global_inds = self.arr.global_from_local(*local_inds)
        return global_inds, value


def ndenumerate(arr):
    return GlobalIterator(arr)


def fromfunction(function, shape, **kwargs):
    dtype = kwargs.pop('dtype', int)
    dist = kwargs.pop('dist', {0: 'b'})
    grid_shape = kwargs.pop('grid_shape', None)
    comm = kwargs.pop('comm', None)
    da = empty(shape, dtype, dist, grid_shape, comm)
    for global_inds, x in ndenumerate(da):
        da[global_inds] = function(*global_inds, **kwargs)
    return da


def fromlocalarray_like(local_arr, like_arr):
    """
    Create a new LocalArray using a given local array (+its dtype).
    """
    res = LocalArray(like_arr.global_shape, local_arr.dtype, like_arr.dist,
                     like_arr.grid_shape, like_arr.base_comm, buf=local_arr)
    return res


def identity(n, dtype=np.intp):
    _raise_nie()


def where(condition, x=None, y=None):
    _raise_nie()


#----------------------------------------------------------------------------
# 4.2 Operations on two or more arrays
#----------------------------------------------------------------------------

def arecompatible(a, b):
    """Do these arrays have the same compatibility hash?"""
    return a.compatibility_hash() == b.compatibility_hash()


def concatenate(seq, axis=0):
    _raise_nie()


def correlate(x, y, mode='valid'):
    _raise_nie()


def convolve(x, y, mode='valid'):
    _raise_nie()


def outer(a, b):
    _raise_nie()


def inner(a, b):
    _raise_nie()


def dot(a, b):
    _raise_nie()


def vdot(a, b):
    _raise_nie()


def tensordot(a, b, axes=(-1, 0)):
    _raise_nie()


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    _raise_nie()


def allclose(a, b, rtol=10e-5, atom=10e-8):
    _raise_nie()


#----------------------------------------------------------------------------
# 4.3 Printing arrays
#----------------------------------------------------------------------------


def distarray2string(a):
    _raise_nie()


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None):
    res = np.set_printoptions(precision, threshold, edgeitems, linewidth,
                              suppress)
    return res


def get_printoptions():
    return np.get_printoptions()


#----------------------------------------------------------------------------
# 4.5 Dealing with data types
#----------------------------------------------------------------------------


dtype = np.dtype
maximum_sctype = np.maximum_sctype
issctype = np.issctype
obj2sctype = np.obj2sctype
sctype2char = np.sctype2char
can_cast = np.can_cast


#----------------------------------------------------------------------------
# 5 Additional convenience routines
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.1 Shape functions
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.2 Basic functions
#----------------------------------------------------------------------------


def sum(a, dtype=None):
    local_sum = a.local_array.sum(dtype=dtype)
    global_sum = a.comm.allreduce(local_sum, None, op=MPI.SUM)
    return global_sum


def average(a, axis=None, weights=None, returned=0):
    _raise_nie()


def cov(x, y=None, rowvar=1, bias=0):
    _raise_nie()


def corrcoef(x, y=None, rowvar=1, bias=0):
    _raise_nie()


def median(m):
    _raise_nie()


def digitize(x, bins):
    _raise_nie()


def histogram(x, bins=None, range=None, normed=False):
    _raise_nie()


def histogram2d(x, y, bins, normed=False):
    _raise_nie()


def logspace(start, stop, num=50, endpoint=True, base=10.0):
    _raise_nie()


def linspace(start, stop, num=50, endpoint=True, retstep=False):
    _raise_nie()


#----------------------------------------------------------------------------
# 5.3 Polynomial functions
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.4 Set operations
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.5 Array construction using index tricks
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.6 Other indexing devices
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.7 Two-dimensional functions
#----------------------------------------------------------------------------


def eye(n, m=None, k=0, dtype=float):
    _raise_nie()


def diag(v, k=0):
    _raise_nie()


#----------------------------------------------------------------------------
# 5.8 More data type functions
#----------------------------------------------------------------------------


issubclass_ = np.issubclass_
issubdtype = np.issubdtype
iscomplexobj = np.iscomplexobj
isrealobj = np.isrealobj
isscalar = np.isscalar
nan_to_num = np.nan_to_num
real_if_close = np.real_if_close
cast = np.cast
mintypecode = np.mintypecode
finfo = np.finfo


#----------------------------------------------------------------------------
# 5.9 Functions that behave like ufuncs
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.10 Misc functions
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# 5.11 Utility functions
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Universal Functions
#
# I would really like these functions to be in a separate file, but that
# is not possible because of circular import problems.  Basically, these
# functions need access to the LocalArray object in this module, and the
# LocalArray object needs to use these functions.  There are 3 options for
# solving this problem:
#
#     * Put everything in one file
#     * Put the functions needed by LocalArray in distarray, others elsewhere
#     * Make a subclass of LocalArray that has methods that use the functions
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


# Functions for manipulating shapes according to the broadcast rules.

def _expand_shape(s, length, element=1):
    add = length - len(s)
    if add > 0:
        return add*(element,)+s
    else:
        return s


def _prepend_ones(*args):
    max_length = max(len(a) for a in args)
    return [_expand_shape(s, max_length, 1) for s in args]


def _prepend_nones(*args):
    max_length = max(len(a) for a in args)
    return [_expand_shape(s, max_length, None) for s in args]


def _return_shape(*args):
    return tuple([max(i) for i in zip(*args)])


def _are_shapes_bcast(shape, target_shape):
    for si, tsi in zip(shape, target_shape):
        if not si == 1 and not si == tsi:
            return False
    return True


class LocalArrayUnaryOperation(object):

    def __init__(self, numpy_ufunc):
        self.func = numpy_ufunc
        self.__doc__ = getattr(numpy_ufunc, "__doc__", str(numpy_ufunc))
        self.__name__ = getattr(numpy_ufunc, "__name__", str(numpy_ufunc))

    def __call__(self, x1, y=None, *args, **kwargs):
        # What types of input are allowed?
        x1_isdla = isinstance(x1, LocalArray)
        y_isdla = isinstance(y, LocalArray)
        assert x1_isdla or isscalar(x1), "Invalid type for unary ufunc"
        assert y is None or y_isdla, "Invalid return array type"
        if y is None:
            return self.func(x1, *args, **kwargs)
        elif y_isdla:
            if x1_isdla:
                if not arecompatible(x1, y):
                    raise IncompatibleArrayError("Incompatible LocalArrays")
            self.func(x1, y.local_array, *args, **kwargs)
            return y
        else:
            raise TypeError("Invalid return type for unary ufunc")

    def __str__(self):
        return "LocalArray version of " + str(self.func)


class LocalArrayBinaryOperation(object):

    def __init__(self, numpy_ufunc):
        self.func = numpy_ufunc
        self.__doc__ = getattr(numpy_ufunc, "__doc__", str(numpy_ufunc))
        self.__name__ = getattr(numpy_ufunc, "__name__", str(numpy_ufunc))

    def __call__(self, x1, x2, y=None, *args, **kwargs):
        # What types of input are allowed?
        x1_isdla = isinstance(x1, LocalArray)
        x2_isdla = isinstance(x2, LocalArray)
        y_isdla = isinstance(y, LocalArray)
        assert x1_isdla or isscalar(x1), "Invalid type for binary ufunc"
        assert x2_isdla or isscalar(x2), "Invalid type for binary ufunc"
        assert y is None or y_isdla
        if y is None:
                if x1_isdla and x2_isdla:
                    if not arecompatible(x1, x2):
                        raise IncompatibleArrayError("Incompatible DistArrays")
                return self.func(x1, x2, *args, **kwargs)
        elif y_isdla:
            if x1_isdla:
                if not arecompatible(x1, y):
                    raise IncompatibleArrayError("Incompatible LocalArrays")
            if x2_isdla:
                if not arecompatible(x2, y):
                    raise IncompatibleArrayError("Incompatible LocalArrays")
            kwargs.pop('y', None)
            self.func(x1, x2, y.local_array, *args, **kwargs)
            return y
        else:
            raise TypeError("Invalid return type for unary ufunc")

    def __str__(self):
        return "LocalArray version of " + str(self.func)


def _add_operations(wrapper, ops):
    """Wrap numpy ufuncs for `LocalArray`s.

    Wraps numpy ufuncs and adds them to this module's namespace.

    Parameters
    ----------
    wrapper : callable
        Takes a numpy ufunc and returns a LocalArray ufunc.
    ops : iterable of callables
        All of the callables to wrap with `wrapper`.
    """
    for op in ops:
        fn_name = "np." + op
        fn_value = wrapper(eval(fn_name))
        names = globals()
        names[op] = fn_value


# numpy unary operations to wrap
_unary_ops = ('absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
              'arctanh', 'conjugate', 'cos', 'cosh', 'exp', 'expm1', 'invert',
              'log', 'log10', 'log1p', 'negative', 'reciprocal', 'rint',
              'sign', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh')

# numpy binary operations to wrap
_binary_ops = ('add', 'arctan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
               'divide', 'floor_divide', 'fmod', 'hypot', 'left_shift', 'mod',
               'multiply', 'power', 'remainder', 'right_shift', 'subtract',
               'true_divide', 'less', 'less_equal', 'equal', 'not_equal',
               'greater', 'greater_equal',)

_add_operations(LocalArrayUnaryOperation, _unary_ops)
_add_operations(LocalArrayBinaryOperation, _binary_ops)
