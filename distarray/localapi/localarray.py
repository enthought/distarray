# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
The `LocalArray` data structure.

`DistArray` objects are proxies for collections of `LocalArray` objects (that
usually reside on engines).
"""

from __future__ import print_function, division


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from collections import Mapping

import numpy as np

from distarray.externals import six
from distarray.externals.six.moves import zip

from distarray.metadata_utils import sanitize_indices

from distarray.localapi.mpiutils import MPI
from distarray.localapi import format, maps
from distarray.localapi.error import InvalidDimensionError, IncompatibleArrayError


class GlobalIndex(object):
    """Object which provides access to global indexing on LocalArrays."""
    def __init__(self, distribution, ndarray):
        self.distribution = distribution
        self.ndarray = ndarray

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

    def _local_from_global(self, global_ind):
        return self.distribution.local_from_global(global_ind)

    def get_slice(self, global_inds, new_distribution):
        try:
            local_inds = self._local_from_global(global_inds)
        except KeyError as err:
            raise IndexError(err)
        view = self.ndarray[local_inds]
        return LocalArray(distribution=new_distribution,
                          dtype=self.ndarray.dtype,
                          buf=view)

    def __getitem__(self, global_inds):
        return_type, global_inds = sanitize_indices(global_inds)
        if return_type == 'view':
            msg = "__getitem__ does not support slices.  See `get_slice`."
            raise TypeError(msg)

        try:
            local_inds = self._local_from_global(global_inds)
        except KeyError as err:
            raise IndexError(err)

        return self.ndarray[local_inds]


    def __setitem__(self, global_inds, value):
        _, global_inds = sanitize_indices(global_inds)
        try:
            local_inds = self._local_from_global(global_inds)
            self.ndarray[local_inds] = value
        except KeyError as err:
            raise IndexError(err)


class LocalArray(object):
    """Distributed memory Python arrays."""

    __array_priority__ = 20.0

    #-------------------------------------------------------------------------
    # Methods used for initialization
    #-------------------------------------------------------------------------

    def __init__(self, distribution, dtype=None, buf=None):
        """Make a LocalArray from a `dim_data` tuple.

        Parameters
        ----------
        distribution : local._maps.Distribution object

        Returns
        -------
        LocalArray
            A LocalArray encapsulating `buf`, or else an empty
            (uninitialized) LocalArray.
        """
        self.distribution = distribution

        # create the buffer
        if buf is None:
            self._ndarray = np.empty(self.local_shape, dtype=dtype)
        else:
            self._ndarray = np.asarray(buf, dtype=dtype)
            if distribution.local_shape != self.ndarray.shape:
                msg = "distribution shape must equal buf shape."
                raise RuntimeError(msg)

        # We pass a view of self.ndarray because we want the
        # GlobalIndex object to be able to change the LocalArray
        # object's data.
        self.global_index = GlobalIndex(self.distribution,
                                        self.ndarray.view())

        self.base = None  # mimic numpy.ndarray.base
        self.ctypes = None  # mimic numpy.ndarray.ctypes

    @property
    def dim_data(self):
        return self.distribution.dim_data

    @property
    def dist(self):
        return self.distribution.dist

    @property
    def global_shape(self):
        return self.distribution.global_shape

    @property
    def ndim(self):
        return self.distribution.ndim

    @property
    def global_size(self):
        return self.distribution.global_size

    @property
    def comm(self):
        return self.distribution.comm

    @property
    def comm_size(self):
        return self.distribution.comm_size

    @property
    def comm_rank(self):
        return self.distribution.comm_rank

    @property
    def cart_coords(self):
        return self.distribution.cart_coords

    @property
    def grid_shape(self):
        return self.distribution.grid_shape

    @property
    def local_shape(self):
        lshape = self.distribution.local_shape
        assert lshape == self.distribution.local_shape
        return lshape

    @property
    def local_size(self):
        lsize = self.distribution.local_size
        assert lsize == self.ndarray.size
        return lsize

    @property
    def local_data(self):
        return self.ndarray.data

    @property
    def dtype(self):
        return self.ndarray.dtype

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.global_size * self.itemsize

    def compatibility_hash(self):
        return hash((self.global_shape, self.dist, self.grid_shape, True))

    #-------------------------------------------------------------------------
    # Distributed Array Protocol
    #-------------------------------------------------------------------------

    @classmethod
    def from_distarray(cls, comm, obj):
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

        distribution = maps.Distribution(comm=comm, dim_data=dim_data)
        return cls(distribution=distribution, buf=buf)

    def __distarray__(self):
        """Returns the data structure required by the DAP.

        DAP = Distributed Array Protocol

        See the project's documentation for the Protocol's specification.
        """
        distbuffer = {
            "__version__": "0.10.0",
            "buffer": self.ndarray,
            "dim_data": self.dim_data,
        }
        return distbuffer

    #-------------------------------------------------------------------------
    # Methods related to distributed indexing
    #-------------------------------------------------------------------------

    def _get_ndarray(self):
        return self._ndarray

    def _set_ndarray(self, a):
        arr = np.asarray(a, dtype=self.dtype, order='C')
        if arr.shape == self.local_shape:
            self._ndarray = arr
        else:
            raise ValueError("Incompatible local array shape")

    ndarray = property(_get_ndarray, _set_ndarray)

    def coords_from_rank(self, rank):
        return self.distribution.coords_from_rank(rank)

    def rank_from_coords(self, coords):
        return self.distribution.rank_from_coords(coords)

    def local_from_global(self, global_ind):
        return self.distribution.local_from_global(global_ind)

    def global_from_local(self, local_ind):
        return self.distribution.global_from_local(local_ind)

    def global_limits(self, dim):
        if dim < 0 or dim >= self.ndim:
            raise InvalidDimensionError("Invalid dimension: %r" % dim)
        lower_local = self.ndim * [0]
        lower_global = self.global_from_local(lower_local)
        upper_local = [shape - 1 for shape in self.local_shape]
        upper_global = self.global_from_local(upper_local)
        return lower_global[dim], upper_global[dim]

    #-------------------------------------------------------------------------
    # ndarray methods
    #-------------------------------------------------------------------------
    # Array conversion
    #-------------------------------------------------------------------------

    def astype(self, newdtype):
        """Return a copy of this LocalArray with a new underlying dtype."""
        if newdtype is None:
            return self.copy()
        else:
            local_copy = self.ndarray.astype(newdtype)
            new_da = self.__class__(distribution=self.distribution,
                                    dtype=newdtype,
                                    buf=local_copy)
            return new_da

    def copy(self):
        """Return a copy of this LocalArray."""
        local_copy = self.ndarray.copy()
        return self.__class__(distribution=self.distribution,
                              dtype=self.dtype,
                              buf=local_copy)

    def local_view(self, dtype=None):
        if dtype is None:
            return self.ndarray.view()
        else:
            return self.ndarray.view(dtype)

    def view(self, distribution, dtype):
        """Return a new LocalArray whose underlying `ndarray` is a view on
        `self.ndarray`.
        """
        return self.__class__(distribution=distribution,
                              dtype=dtype,
                              buf=self.local_view(dtype=dtype))

    def __array__(self, dtype=None):
        if dtype is None:
            return self.ndarray
        elif np.dtype(dtype) == self.dtype:
            return self.ndarray
        else:
            return self.ndarray.astype(dtype)

    def __array_wrap__(self, obj, context=None):
        """
        Return a LocalArray based on obj.

        This method constructs a new LocalArray object using the distribution
        from self and the buffer from obj.

        This is used to construct return arrays for ufuncs.
        """
        return self.__class__(self.distribution, buf=obj)

    def fill(self, scalar):
        self.ndarray.fill(scalar)

    def asdist_like(self, other):
        """
        Return a version of self that has shape, dist and grid_shape like
        `other`.
        """
        if arecompatible(self, other):
            return self
        else:
            raise IncompatibleArrayError("DistArrays have incompatible shape,"
                                         "dist or grid_shape")

    #-------------------------------------------------------------------------
    # Basic customization
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
        return str(self.ndarray)

    def __repr__(self):
        return str(self.ndarray)

    #-------------------------------------------------------------------------
    # Container customization
    #-------------------------------------------------------------------------

    def __len__(self):
        return self.global_shape[0]

    def __getitem__(self, index):
        """Get a local item."""
        return_type, index = sanitize_indices(index)
        if return_type == 'value':
            return self.ndarray[index]
        elif return_type == 'view':
            msg = "__getitem__ does not support slices.  See `global_index.get_item`."
            raise TypeError(msg)
        else:
            assert False  # impossible is nothing

    def __setitem__(self, index, value):
        """Set a local item."""
        self.ndarray[index] = value

    def sync(self):
        raise NotImplementedError("`sync` not yet implemented.")

    def __contains__(self, item):
        return item in self.ndarray

    def pack_index(self, inds):
        inds_array = np.array(inds)
        strides_array = np.cumprod([1] + list(self.global_shape)[:0:-1])[::-1]
        return np.sum(inds_array * strides_array)

    def unpack_index(self, packed_ind):
        if packed_ind > np.prod(self.global_shape) - 1 or packed_ind < 0:
            raise ValueError("Invalid index, must be 0 <= x <= number of"
                             "elements.")
        strides_array = np.cumprod([1] + list(self.global_shape)[:0:-1])[::-1]
        return tuple(packed_ind // strides_array % self.global_shape)

    #--------------------------------------------------------------------------
    # Arithmetic customization - binary
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

    # Unary

    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __invert__(self):
        return invert(self)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Creating arrays
# ---------------------------------------------------------------------------

def empty(distribution, dtype=float):
    """Create an empty LocalArray."""
    return LocalArray(distribution=distribution, dtype=dtype)


def empty_like(arr, dtype=None):
    """Create an empty LocalArray with a distribution like `arr`."""
    if isinstance(arr, LocalArray):
        if dtype is None:
            return empty(distribution=arr.distribution, dtype=arr.dtype)
        else:
            return empty(distribution=arr.distribution, dtype=dtype)
    else:
        raise TypeError("A LocalArray or subclass is expected")


def zeros(distribution, dtype=float):
    """Create a LocalArray filled with zeros."""
    la = LocalArray(distribution=distribution, dtype=dtype)
    la.fill(0)
    return la


def zeros_like(arr, dtype=float):
    """Create a LocalArray of zeros with a distribution like `arr`."""
    if isinstance(arr, LocalArray):
        if dtype is None:
            return zeros(distribution=arr.distribution, dtype=arr.dtype)
        else:
            return zeros(distribution=arr.distribution, dtype=dtype)
    else:
        raise TypeError("A LocalArray or subclass is expected")


def ones(distribution, dtype=float):
    """Create a LocalArray filled with ones."""
    la = LocalArray(distribution=distribution, dtype=dtype)
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


def load_dnpy(comm, file):
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
        return LocalArray.from_distarray(comm=comm, obj=distbuffer)

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

    with h5py.File(filename, mode, driver='mpio',
                   comm=arr.distribution.comm) as fp:
        dset = fp.create_dataset(key, arr.global_shape, dtype=arr.dtype)
        try:
            gslice = arr.distribution.global_slice
            dset[gslice] = arr
        except AttributeError:
            # can't represent index with a slice; do it the slow way
            for index, value in ndenumerate(arr):
                dset[index] = value


def compact_indices(dim_data):
    """Given a `dim_data` structure, return a tuple of compact indices.

    For every dimension in `dim_data`, return a representation of the indices
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
            return list(maps.map_from_dim_dict(dd).global_iter)

    def unstructured_index(dd):
        return list(maps.map_from_dim_dict(dd).global_iter)

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


def load_hdf5(comm, filename, dim_data, key='buffer'):
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
    comm : MPI comm object
    key : str, optional
        The identifier for the group to load the LocalArray from (the default
        is 'buffer').

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

    distribution = maps.Distribution(comm=comm, dim_data=dim_data)
    return LocalArray(distribution=distribution, dtype=dtype, buf=buf)


def load_npy(comm, filename, dim_data):
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
    comm : MPI comm object

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
    distribution = maps.Distribution(comm=comm, dim_data=dim_data)
    return LocalArray(distribution=distribution, dtype=data.dtype, buf=buf)


class GlobalIterator(six.Iterator):
    def __init__(self, arr):
        self.arr = arr
        self.nditerator = np.ndenumerate(self.arr.local_view())

    def __iter__(self):
        return self

    def __next__(self):
        local_inds, value = six.advance_iterator(self.nditerator)
        global_inds = self.arr.global_from_local(local_inds)
        return global_inds, value


def ndenumerate(arr):
    return GlobalIterator(arr)


def fromfunction(function, distribution, **kwargs):
    dtype = kwargs.pop('dtype', float)
    da = empty(distribution=distribution, dtype=dtype)
    for global_inds, x in ndenumerate(da):
        da.global_index[global_inds] = function(*global_inds, **kwargs)
    return da


def fromndarray_like(ndarray, like_arr):
    """Create a new LocalArray like `like_arr` with buffer set to `ndarray`.
    """
    return LocalArray(like_arr.distribution, buf=ndarray)

# ---------------------------------------------------------------------------
# Operations on two or more arrays
# ---------------------------------------------------------------------------

def arecompatible(a, b):
    """Do these arrays have the same compatibility hash?"""
    return a.compatibility_hash() == b.compatibility_hash()


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None):
    np.set_printoptions(precision, threshold, edgeitems, linewidth, suppress)


def get_printoptions():
    return np.get_printoptions()


# ---------------------------------------------------------------------------
# Dealing with data types
# ---------------------------------------------------------------------------


dtype = np.dtype
maximum_sctype = np.maximum_sctype
issctype = np.issctype
obj2sctype = np.obj2sctype
sctype2char = np.sctype2char
can_cast = np.can_cast


# ---------------------------------------------------------------------------
# Reduction functions
# ---------------------------------------------------------------------------

def local_reduction(out_comm, reducer, larr, ddpr, dtype, axes):
    """ Entry point for reductions on local arrays.

    Parameters
    ----------
    reducer : callable
        Performs the core reduction operation.

    out_comm: MPI Comm instance.
        The MPI communicator for the result of the reduction.  Is equal to
        MPI.COMM_NULL when this rank is not part of the output communicator.

    larr: LocalArray
        Input.  Defined for all ranks.

    ddpr: sequence of dim-data dictionaries.

    axes: Sequence of ints or None.

    Returns
    -------
    LocalArray or None
        When out_comm == MPI.COMM_NULL, returns None.
        Otherwise, returns the LocalArray section of the reduction result.
    """

    if out_comm == MPI.COMM_NULL:
        out = None
    else:
        dim_data = ddpr[out_comm.Get_rank()] if ddpr else ()
        dist = maps.Distribution(comm=out_comm, dim_data=dim_data)
        out = empty(dist, dtype)

    remaining_dims = [False] * larr.ndim
    for axis in axes:
        remaining_dims[axis] = True
    reduce_comm = larr.comm.Sub(remaining_dims)
    return reducer(reduce_comm, larr, out, axes, dtype)

# --- Reductions for min, max, sum, mean, var, std ----------------------------

def _basic_reducer(reduce_comm, op, func, args, kwargs, out):
    """ Handles simple reductions: min, max, sum.  Internal. """
    if out is None:
        out_ndarray = None
    else:
        out_ndarray = out.ndarray
        if out.ndarray.dtype == np.bool:
            out.ndarray.dtype = np.uint8
    # Use asarray() to coerce np scalars to zero-dimensional arrays.
    local_reduce = np.asarray(func(*args, **kwargs))
    reduce_comm.Reduce(local_reduce, out_ndarray, op=op, root=0)
    return out


def min_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for min."""
    if larr.ndarray.dtype == np.bool:
        larr.ndarray.dtype = np.uint8
    return _basic_reducer(reduce_comm, MPI.MIN,
                          larr.ndarray.min,
                          (), {'axis': axes}, out)


def max_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for max."""
    if larr.ndarray.dtype == np.bool:
        larr.ndarray.dtype = np.uint8
    return _basic_reducer(reduce_comm, MPI.MAX,
                          larr.ndarray.max,
                          (), {'axis': axes}, out)


def sum_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for sum."""
    if larr.ndarray.dtype == np.bool:
        larr.ndarray.dtype = np.uint8
    return _basic_reducer(reduce_comm, MPI.SUM,
                          larr.ndarray.sum,
                          (), {'axis': axes, 'dtype': dtype}, out)


def mean_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for mean."""
    sum_reducer(reduce_comm, larr, out, axes, dtype)
    if out is not None:
        out.ndarray /= (larr.global_size / out.global_size)
    return out


def var_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for var."""
    temp = empty_like(larr, dtype=float)

    # We hold the intermediate means in `mean`.
    mean = empty_like(out, dtype=float) if out is not None else None
    mean = mean_reducer(reduce_comm, larr, mean, axes, dtype=float)

    # Have to broadcast mean.ndarray to all ranks in this reduce_comm.
    mean = reduce_comm.bcast(mean, root=0)

    mean_shape = []
    for (ax, s) in enumerate(larr.ndarray.shape):
        mean_shape.append(1 if ax in axes else s)
    mean_shape = tuple(mean_shape)

    mean.ndarray.shape = mean_shape

    # Do the variance calculation.
    temp.ndarray[...] = (larr.ndarray - mean.ndarray) ** 2

    # Get the mean reduction of temp's data.
    mean_reducer(reduce_comm, temp, out, axes, dtype)
    return out


def std_reducer(reduce_comm, larr, out, axes, dtype):
    """ Core reduction function for std."""
    var_reducer(reduce_comm, larr, out, axes, dtype)
    if out is not None:
        np.sqrt(out.ndarray, out=out.ndarray)
    return out

# ---------------------------------------------------------------------------
# More data type functions
# ---------------------------------------------------------------------------


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

# Functions for manipulating shapes according to the broadcast rules.

def _expand_shape(s, length, element=1):
    add = length - len(s)
    if add > 0:
        return add * (element,) + s
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
            self.func(x1, y.ndarray, *args, **kwargs)
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
            self.func(x1, x2, y.ndarray, *args, **kwargs)
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
