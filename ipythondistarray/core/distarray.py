#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import sys
import math

import numpy as np

from ipythondistarray.mpi import mpibase
from ipythondistarray.mpi.mpibase import MPI
from ipythondistarray.core import maps
from ipythondistarray.core.error import *
from ipythondistarray.core.base import BaseDistArray, arecompatible
from ipythondistarray.core.construct import (
    init_base_comm,
    init_dist,
    init_distdims,
    init_map_classes,
    init_grid_shape,
    optimize_grid_shape,
    init_comm,
    init_local_shape_and_maps,
    find_local_shape,
    find_grid_shape)
from ipythondistarray.utils import _raise_nie


#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------


__all__ = [
    'DistArray',
    'empty',
    'empty_like',
    'zeros',
    'zeros_like',
    'ones',
    'fromfunction',
    'set_printoptions',
    'get_printoptions',
    'dtype',
    'maximum_sctype',
    'issctype',
    'obj2sctype',
    'sctype2char',
    'can_cast',
    'issubclass_',
    'issubdtype',
    'iscomplexobj',
    'isrealobj',
    'isscalar',
    'nan_to_num',
    'real_if_close',
    'cast',
    'mintypecode',
    'finfo',
    'sum',
    'add',
    'subtract',
    'divide',
    'true_divide',
    'floor_divide',
    'power',
    'remainder',
    'fmod',
    'arctan2',
    'hypot',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'negative',
    'absolute',
    'rint',
    'sign',
    'conjugate',
    'exp',
    'log',
    'expm1',
    'log1p',
    'log10',
    'sqrt',
    'square',
    'reciprocal',
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'invert']

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Base DistArray class
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


class DenseDistArray(BaseDistArray):
    """Distribute memory Python arrays."""
    
    __array_priority__ = 20.0
    
    def __init__(self, shape, dtype=float, dist={0:'b'} , grid_shape=None,
                 comm=None, buf=None, offset=0):
        """Create a distributed memory array on a set of processors.
        """
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(dtype)
        self.size = reduce(lambda x,y: x*y, shape)
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size*self.itemsize
        self.data = None
        self.base = None
        self.ctypes = None
        
        # This order is extremely important and is shown by the arguments passed on to
        # subsequent _init_* methods.  It is critical that these _init_* methods are free
        # of side effects and stateless.  This means that they cannot set or get class or
        # instance attributes
        self.base_comm = init_base_comm(comm)
        self.comm_size = self.base_comm.Get_size()
        self.comm_rank = self.base_comm.Get_rank()
        
        self.dist = init_dist(dist, self.ndim)
        self.distdims = init_distdims(self.dist, self.ndim)
        self.ndistdim = len(self.distdims)
        self.map_classes = init_map_classes(self.dist)
        
        self.grid_shape = init_grid_shape(self.shape, grid_shape, 
            self.distdims, self.comm_size)
        self.comm = init_comm(self.base_comm, self.grid_shape, self.ndistdim)
        self.cart_coords = self.comm.Get_coords(self.comm_rank)
        self.local_shape, self.maps = init_local_shape_and_maps(self.shape, 
            self.grid_shape, self.distdims, self.map_classes)
        self.local_size = reduce(lambda x,y: x*y, self.local_shape)
        
        # At this point, everything is setup, but the memory has not been allocated.
        self._allocate(buf, offset)
    
    def __del__(self):
        if self.comm is not None:
            self.comm.Free()
    
    def compatibility_hash(self):
        return hash((self.shape, self.dist, self.grid_shape, True))
    
    #----------------------------------------------------------------------------
    # Misc methods
    #---------------------------------------------------------------------------- 
    
    #----------------------------------------------------------------------------
    # Methods used at initialization
    #----------------------------------------------------------------------------   
    
    def _allocate(self, buf=None, offset=0):
        if buf is None:
            # Allocate a new array and use its data attribute as my own
            self.local_array = np.empty(self.local_shape, dtype=self.dtype)
            self.data = self.local_array.data
        else:
            try:
                buf = buffer(buf)
            except TypeError:
                raise TypeError("the object is not or can't be made into a buffer")
            try:
                self.local_array = np.frombuffer(buf, dtype=self.dtype, count=self.local_size, offset=offset)
                self.local_array.shape = self.local_shape
                self.data = self.local_array.data
            except ValueError:
                raise ValueError("the buffer is smaller than needed for this array")
    
    #----------------------------------------------------------------------------
    # Methods related to distributed indexing
    #----------------------------------------------------------------------------   
    
    def get_localarray(self):
        return self.local_view()
    
    def set_localarray(self, a):
        a = np.asarray(a, dtype=self.dtype, order='C')
        if a.shape != self.local_shape:
            raise ValueError("incompatible local array shape")
        b = buffer(a)
        self.local_array = np.frombuffer(b,dtype=self.dtype)
        self.local_array.shape = self.local_shape
    
    def owner_rank(self, *indices):
        owners = [self.maps[i].owner(indices[self.distdims[i]]) for i in range(self.ndistdim)]
        return self.comm.Get_cart_rank(owners)
    
    def owner_coords(self, *indices):
        owners = [self.maps[i].owner(indices[self.distdims[i]]) for i in range(self.ndistdim)]
        return owners          
    
    def rank_to_coords(self, rank):
        return self.comm.Get_coords(rank)
    
    def coords_to_rank(self, coords):
        return self.comm.Get_cart_rank(coords)
    
    def local_ind(self, *global_ind):
        local_ind = list(global_ind)
        for i in range(self.ndistdim):
            dd = self.distdims[i]
            local_ind[dd] = self.maps[i].local_index(global_ind[dd])
        return tuple(local_ind)
    
    def global_ind(self, owner, *local_ind):
        if isinstance(owner, int):
            owner_coords = self.rank_to_coords(owner)
        else:
            owner_coords = owner
        global_ind = list(local_ind)
        for i in range(self.ndistdim):
            dd = self.distdims[i]
            global_ind[dd] = self.maps[i].global_index(owner_coords[i], local_ind[dd])
        return tuple(global_ind)
    
    def get_dist_matrix(self):
        if self.ndim==2:
            a = np.empty(self.shape,dtype=int)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    a[i,j] = self.owner_rank(i,j)
            return a
        else:
            raise DistMatrixError("The dist matrix can only be created for a 2d array")        
    
    def plot_dist_matrix(self):
        try:
            dm = self.get_dist_matrix()
        except DistMatrixError:
            pass
        else:
            if self.comm_rank==0:
                try:
                    import pylab
                except ImportError:
                    print "Matplotlib is not installed so the dist_matrix cannot be plotted"
                else:
                    pylab.ion()
                    pylab.matshow(dm)
                    pylab.colorbar()
                    pylab.xlabel('columns')
                    pylab.ylabel('rows')
                    pylab.title('Memory Distribution Plot')
                    pylab.draw() 
                    pylab.show()
                    
    
    #----------------------------------------------------------------------------
    # 3.2 ndarray methods
    #----------------------------------------------------------------------------   
    
    #----------------------------------------------------------------------------
    # 3.2.1 Array conversion
    #---------------------------------------------------------------------------- 
    
    def astype(self, dtype):
        if dtype is None:
            return self.copy()
        else:
            local_copy = self.local_array.astype(dtype)
            new_da = DistArray(self.shape, dtype=self.dtype, dist=self.dist,
                grid_shape=self.grid_shape, comm=self.base_comm, buf=local_copy)
            return new_da
    
    def copy(self):
        local_copy = self.local_array.copy()
        new_da = DistArray(self.shape, dtype=self.dtype, dist=self.dist,
            grid_shape=self.grid_shape, comm=self.base_comm, buf=local_copy)
    
    def local_view(self, dtype=None):
        if dtype is None:
            return self.local_array.view()
        else:
            return self.local_array.view(dtype)
            
    def view(self, dtype=None):
        if dtype is None:
            new_da = DistArray(self.shape, self.dtype, self.dist,
                self.grid_shape, self.base_comm, buf=self.data)
        else:
            new_da = DistArray(self.shape, dtype, self.dist,
                self.grid_shape, self.base_comm, buf=self.data)
        return new_da
    
    def __distarray__(self, dtype=None):
        return self
    
    def __array__(self, dtype=None):
        if dtype is None:
            return self.local_array
        elif np.dtype(dtype)==self.dtype:
            return self.local_array
        else:
            return self.local_array.astype(dtype)
    
    def __array_wrap__(self, obj, context=None):
        """
        Return a DistArray based on obj.
        
        This method constructs a new DistArray object using (shape, dist,
        grid_shape and base_comm) from self and dtype, buffer from obj.
        
        This is used to construct return arrays for ufuncs.
        """
        return DistArray(self.shape, obj.dtype, self.dist, self.grid_shape, 
            self.base_comm, buf=obj)
    
    
    def fill(self, scalar):
        self.local_array.fill(scalar)
    
    #----------------------------------------------------------------------------
    # 3.2.2 Array shape manipulation
    #---------------------------------------------------------------------------- 
    
    def reshape(self, newshape):
        _raise_nie()
    
    def redist(self, newshape, newdist={0:'b'}, newgrid_shape=None):
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
     
    def asdist(self, shape, dist={0:'b'}, grid_shape=None):
        pass
        # new_da = DistArray(shape, self.dtype, dist, grid_shape, self.base_comm)
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
        #     global_inds = self.global_ind(new_da.comm_rank, old_local_inds)
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
        #         new_local_ind = local_ind_by_owner_and_location(old_owner, location)
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
        Return a version of self that has shape, dist and grid_shape like other.
        """
        if arecompatible(self, other):
            return self
        else:
            raise IncompatibleArrayError("DistArrays have incompatible shape, dist or grid_shape")
    
    #----------------------------------------------------------------------------
    # 3.2.3 Array item selection and manipulation
    #----------------------------------------------------------------------------   
    
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
    
    #----------------------------------------------------------------------------
    # 3.2.4 Array item selection and manipulation
    #---------------------------------------------------------------------------- 
    
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
    
    congugate = conj
    
    def round(self, decimals=0, out=None):
        _raise_nie()
    
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        _raise_nie()
    
    # def sum(self, axis=None, dtype=None, out=None):    
    def sum(self, axis=None, dtype=None, out=None):
        return sum(self, dtype)
    
    def cumsum(self, axis=None, dtype=None, out=None):        
        _raise_nie()
    
    def mean(self, axis=None, dtype=None, out=None):
        return self.sum(dtype=dtype)/self.size
    
    def var(self, axis=None, dtype=None, out=None):
        mu = self.mean()
        temp = (self - mu)**2
        return temp.mean()
     
    def std(self, axis=None, dtype=None, out=None):
        return math.sqrt(self.var())
    
    def prod(self, axis=None, dtype=None, out=None):
        _raise_nie()
    
    def cumprod(self, axis=None, dtype=None, out=None):
        _raise_nie()
    
    def all(self, axis=None, out=None):
        _raise_nie()
    
    def any(self, axis=None, out=None):    
        _raise_nie()
    
    #----------------------------------------------------------------------------
    # 3.3 Array special methods
    #---------------------------------------------------------------------------- 
    
    #----------------------------------------------------------------------------
    # 3.3.1 Methods for standard library functions
    #----------------------------------------------------------------------------
    
    def __copy__(self):
        _raise_nie()
    
    def __deepcopy__(self):
        _raise_nie()
    
    #----------------------------------------------------------------------------
    # 3.3.2 Basic customization
    #----------------------------------------------------------------------------
    
    def __lt__(self, other):
        _raise_nie()
    
    def __le__(self, other):
        _raise_nie()
    
    def __gt__(self, other):
        _raise_nie()
    
    def __ge__(self, other):
        _raise_nie()
    
    def __eq__(self, other):
        _raise_nie()
    
    def __ne__(self, other):
        _raise_nie()
    
    def __str__(self):
        return str(self.local_array)
    
    def __repr__(self):
        return str(self.local_array)
    
    def __nonzero__(self):
        _raise_nie()
    
    #----------------------------------------------------------------------------
    # 3.3.3 Container customization
    #----------------------------------------------------------------------------    
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, key):
        _raise_nie()
    
    def __setitem__(self, key, value):
        _raise_nie()
    
    def __contains__(self, item):
        _raise_nie()
    
    #----------------------------------------------------------------------------
    # 3.3.4 Arithmetic customization - binary
    #---------------------------------------------------------------------------- 
    
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
        return self._binary_op_from_ufunc(other, binary_or, '__ror__')
    
    def __xor__(self, other):
        return self._binary_op_from_ufunc(other, binary_xor, '__rxor__')
        
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


def DistArray(shape, dtype=float, dist={0:'b'} , grid_shape=None, comm=None, buf=None, offset=0):
    """
    Create a DistArray of the correct type.
    """
    if comm==MPI.COMM_NULL:
        raise NullCommError("cannot create a DistArray with COMM_NULL")
    else:
        return DenseDistArray(shape, dtype, dist, grid_shape,
            comm, buf, offset)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Functions that are friends of DistArray
#
# I would really like these functions to be in a separate file, but that
# is not possible because of circular import problems.  Basically, these
# functions need accees to the DistArray object in this module, and the 
# DistArray object needs to use these functions.  There are 3 options for
# solving this problem:
# 
#     * Put everything in one file
#     * Put the functions needed by DistArray in distarray, others elsewhere
#     * Make a subclass of DistArray that has methods that use the functions
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

# Here is DistArray.__init__ for reference
# def __init__(self, shape, dtype=float, dist={0:'b'} , grid_shape=None,
#              comm=None, buf=None, offset=0):


def distarray(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    _raise_nie()

    
def asdistarray(object, dtype=None, order=None):
    _raise_nie()

    
def arange(start, stop=None, step=1, dtype=None, dist={0:'b'}, 
    grid_shape=None, comm=None):
    _raise_nie()


def empty(shape, dtype=float, dist={0:'b'}, grid_shape=None, comm=None):
    return DistArray(shape, dtype, dist, grid_shape, comm)


def empty_like(arr):
    if isinstance(arr, DenseDistArray):
        return empty(arr.shape, arr.dtype, arr.dist, arr.grid_shape, arr.base_comm)        
    else:
        raise TypeError("a DenseDistArray or subclass is expected")


def zeros(shape, dtype=float, dist={0:'b'}, grid_shape=None, comm=None):
    base_comm = init_base_comm(comm)
    local_shape = find_local_shape(shape, dist, grid_shape, base_comm.Get_size())
    local_zeros = np.zeros(local_shape, dtype=dtype)
    return DistArray(shape, dtype, dist, grid_shape, comm, buf=local_zeros)


def zeros_like(arr):
    if isinstance(arr, DenseDistArray):
        return zeros(arr.shape, arr.dtype, arr.dist, arr.grid_shape, arr.base_comm)
    else:
        raise TypeError("a DenseDistArray or subclass is expected")


def ones(shape, dtype=float, dist={0:'b'}, grid_shape=None, comm=None):
    base_comm = init_base_comm(comm)
    local_shape = find_local_shape(shape, dist, grid_shape, base_comm.Get_size())
    local_ones = np.ones(local_shape, dtype=dtype)
    return DistArray(shape, dtype, dist, grid_shape, comm, buf=local_ones)


def fromfunction(function, shape, **kwargs):
    dtype = kwargs.pop('dtype', int)
    dist = kwargs.pop('dist', {0:'b'})
    grid_shape = kwargs.pop('grid_shape', None)
    comm = kwargs.pop('comm', None)
    da = empty(shape, dtype, dist, grid_shape, comm)
    local_view = da.local_view()
    for local_inds, x in np.ndenumerate(local_view):
        global_inds = da.global_ind(da.comm_rank, *local_inds)
        local_view[local_inds] = function(*global_inds, **kwargs)
    return da


def fromlocalarray_like(local_arr, like_arr):
    """
    Create a new DistArray using a given local array (+its dtype).
    """
    return DistArray(like_arr.shape, local_arr.dtype, like_arr.dist, like_arr.grid_shape, 
        like_arr.base_comm, buf=local_arr)


def identity(n, dtype=np.intp):
    _raise_nie()


def where(condition, x=None, y=None):
    _raise_nie()


#----------------------------------------------------------------------------
# 4.2 Operations on two or more arrays 
#----------------------------------------------------------------------------


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


def tensordot(a, b, axes=(-1,0)):
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
    return np.set_printoptions(precision, threshold, edgeitems, linewidth, suppress)


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
    local_sum = a.local_array.sum(dtype)
    global_sum = a.comm.Allreduce(local_sum, op=MPI.SUM)
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
# functions need accees to the DistArray object in this module, and the 
# DistArray object needs to use these functions.  There are 3 options for
# solving this problem:
# 
#     * Put everything in one file
#     * Put the functions needed by DistArray in distarray, others elsewhere
#     * Make a subclass of DistArray that has methods that use the functions
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


# Functions for manpulating shapes according to the broadcast rules.

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
        if not si == 1 and not si==tsi:
            return False
    return True


class DistArrayUnaryOperation(object):
    
    def __init__(self, numpy_ufunc):
        self.func = numpy_ufunc
        self.__doc__ = getattr(numpy_ufunc, "__doc__", str(numpy_ufunc))
        self.__name__ = getattr(numpy_ufunc, "__name__", str(numpy_ufunc))
        
    def __call__(self, x1, y=None):
        # What types of input are allowed?
        x1_isdda = isinstance(x1, DenseDistArray)
        y_isdda = isinstance(y, DenseDistArray)
        assert x1_isdda or isscalar(x1), "invalid type for unary ufunc"
        assert y is None or y_isdda, "invalid return array type"
        if y is None:
            return self.func(x1)
        elif y_isdda:
            if x1_isdda:
                if not arecompatible(x1, y):
                    raise IncompatibleArrayError("return DistArray not compatible with DistArray argument" % y)
            local_result = self.func(x1, y.local_array)
            return y
        else:
            raise TypeError("invalid return type for unary ufunc")
    
    def __str__(self):
        return "DistArray version of " + str(self.func)


class DistArrayBinaryOperation(object):
    
    def __init__(self, numpy_ufunc):
        self.func = numpy_ufunc
        self.__doc__ = getattr(numpy_ufunc, "__doc__", str(numpy_ufunc))
        self.__name__ = getattr(numpy_ufunc, "__name__", str(numpy_ufunc))
    
    def __call__(self, x1, x2, y=None):
        # What types of input are allowed?
        x1_isdda = isinstance(x1, DenseDistArray)
        x2_isdda = isinstance(x2, DenseDistArray)
        y_isdda = isinstance(y, DenseDistArray)
        assert x1_isdda or isscalar(x1), "invalid type for binary ufunc"
        assert x2_isdda or isscalar(x2), "invalid type for binary ufunc"
        assert y is None or y_isdda
        if y is None:
                if x1_isdda and x2_isdda:
                    if not arecompatible(x1, x2):
                        raise IncompatibleArrayError("incompatible DistArrays")
                return self.func(x1, x2)
        elif y_isdda:
            if x1_isdda:
                if not arecompatible(x1, y):
                    raise IncompatibleArrayError("incompatible DistArrays")
            if x2_isdda:
                if not arecompatible(x2, y):
                    raise IncompatibleArrayError("incompatible DistArrays")
            local_result = self.func(x1, x2, y.local_array)
            return y
        else:
            raise TypeError("invalid return type for unary ufunc")
    
    def __str__(self):
        return "DistArray version of " + str(self.func)


add = DistArrayBinaryOperation(np.add)
subtract = DistArrayBinaryOperation(np.subtract)
multiply = DistArrayBinaryOperation(np.multiply)
divide = DistArrayBinaryOperation(np.divide)
true_divide = DistArrayBinaryOperation(np.true_divide)
floor_divide = DistArrayBinaryOperation(np.floor_divide)
power = DistArrayBinaryOperation(np.power)
remainder = DistArrayBinaryOperation(np.remainder)
fmod = DistArrayBinaryOperation(np.fmod)
arctan2 = DistArrayBinaryOperation(np.arctan2)
hypot = DistArrayBinaryOperation(np.hypot)
bitwise_and = DistArrayBinaryOperation(np.bitwise_and)
bitwise_or = DistArrayBinaryOperation(np.bitwise_or)
bitwise_xor = DistArrayBinaryOperation(np.bitwise_xor)
left_shift = DistArrayBinaryOperation(np.left_shift)
right_shift = DistArrayBinaryOperation(np.right_shift)


negative = DistArrayUnaryOperation(np.negative)
absolute = DistArrayUnaryOperation(np.absolute)
rint = DistArrayUnaryOperation(np.rint)
sign = DistArrayUnaryOperation(np.sign)
conjugate = DistArrayUnaryOperation(np.conjugate)
exp = DistArrayUnaryOperation(np.exp)
log = DistArrayUnaryOperation(np.log)
expm1 = DistArrayUnaryOperation(np.expm1)
log1p = DistArrayUnaryOperation(np.log1p)
log10 = DistArrayUnaryOperation(np.log10)
sqrt = DistArrayUnaryOperation(np.sqrt)
square = DistArrayUnaryOperation(np.square)
reciprocal = DistArrayUnaryOperation(np.reciprocal)
sin = DistArrayUnaryOperation(np.sin)
cos = DistArrayUnaryOperation(np.cos)
tan = DistArrayUnaryOperation(np.tan)
arcsin = DistArrayUnaryOperation(np.arcsin)
arccos = DistArrayUnaryOperation(np.arccos)
arctan = DistArrayUnaryOperation(np.arctan)
sinh = DistArrayUnaryOperation(np.sinh)
cosh = DistArrayUnaryOperation(np.cosh)
tanh = DistArrayUnaryOperation(np.tanh)
arcsinh = DistArrayUnaryOperation(np.arcsinh)
arccosh = DistArrayUnaryOperation(np.arccosh)
arctanh = DistArrayUnaryOperation(np.arctanh)
invert = DistArrayUnaryOperation(np.invert)


