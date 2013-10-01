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

import sys
import math

import numpy as np

from distarray.mpi import mpibase
from distarray.mpi.mpibase import MPI
from distarray.core.error import *
from distarray.core.construct import (
    init_base_comm,
    init_dist,
    init_distdims,
    init_map_classes,
    init_grid_shape,
    optimize_grid_shape,
    init_comm,
    init_local_shape_and_maps,
    find_local_shape,
    find_grid_shape
)

#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------

__all__ = ['BaseDistArray',
    'arecompatible']

#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------


class BaseDistArray(object):
    """Distribute memory Python arrays."""
    
    __array_priority__ = 20.0
    
    def __init__(self, shape, dtype=float, dist={0:'b'} , grid_shape=None,
                 comm=None, buf=None, offset=0):
        """Create a distributed memory array on a set of processors.
        """
        if comm==MPI.COMM_NULL:
            raise NullCommError("cannot create a LocalArray with COMM_NULL")
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
    
    def __del__(self):
        # If the __init__ method fails, we may not have a valid comm attribute
        # and this needs to be protected against.
        if hasattr(self, 'comm'):
            if self.comm is not None:
                try:
                    self.comm.Free()
                except:
                    pass
    
    def compatibility_hash(self):
        return hash((self.shape, self.dist, self.grid_shape, True))


def arecompatible(a, b):
    """
    Do these arrays have the same compatibility hash?
    """
    return a.compatibility_hash() == b.compatibility_hash()
    