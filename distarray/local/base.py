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

import numpy as np
from functools import reduce

from distarray.mpiutils import MPI
from distarray.local.error import NullCommError
from distarray.local.construct import (init_base_comm, init_dist, init_distdims,
                                      init_map_classes, init_grid_shape,
                                      init_comm, init_local_shape_and_maps)


class BaseLocalArray(object):

    """Distributed memory Python arrays."""

    __array_priority__ = 20.0

    def __init__(self, dimdata, dtype=None, buf=None, comm=None):
        """Make a BaseLocalArray from a `dimdata` tuple.

        Parameters
        ----------
        dimdata : tuple of dictionaries
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
        comm : MPI communicator object, optional

        Returns
        -------
        BaseLocalArray
            A BaseLocalArray encapsulating `buf`, or else an empty
            (uninitialized) BaseLocalArray.
        """
        comm = init_base_comm(comm)

        if comm == MPI.COMM_NULL:
            raise NullCommError("Cannot create a LocalArray with COMM_NULL")

        self.base_comm = init_base_comm(comm)
        self.comm_size = self.base_comm.Get_size()
        self.comm_rank = self.base_comm.Get_rank()

        self.shape = tuple(dd['datasize'] for dd in dimdata)
        self.ndim = len(self.shape)
        self.size = reduce(lambda x,y: x*y, self.shape)

        self.base = None
        self.ctypes = None

        # This order is extremely important and is shown by the
        # arguments passed on to subsequent _init_* methods.  It is
        # critical that these _init_* methods are free of side effects
        # and stateless.  This means that they cannot set or get class
        # or instance attributes.
        dist = {i: dd['disttype'] for (i, dd) in enumerate(dimdata)
                if dd['disttype']}
        self.dist = init_dist(dist, self.ndim)
        self.distdims = init_distdims(self.dist, self.ndim)
        self.ndistdim = len(self.distdims)
        self.map_classes = init_map_classes(self.dist)

        try:
            grid_shape = tuple(
                dd['gridsize'] for dd in dimdata if dd['disttype'])
        except KeyError:
            grid_shape = None
        self.grid_shape = init_grid_shape(self.shape, self.distdims,
                                          self.comm_size, grid_shape)

        self.comm = init_comm(self.base_comm, self.grid_shape, self.ndistdim)
        self.cart_coords = self.comm.Get_coords(self.comm_rank)
        self.local_shape, self.maps = \
            init_local_shape_and_maps(self.shape, self.grid_shape,
                                      self.distdims, self.map_classes)
        self.local_size = reduce(lambda x,y: x*y, self.local_shape)

        if buf is None:
            self.local_array = np.empty(self.local_shape, dtype=dtype)
        else:
            try:
                buf = memoryview(buf)
            except TypeError:
                msg = "The object is not or can't be made into a buffer."
                raise TypeError(msg)
            try:
                self.local_array = np.asarray(buf, dtype=dtype)
            except ValueError:
                msg = "The buffer is smaller than needed for this array."
                raise ValueError(msg)

        self.data = self.local_array.data
        self.dtype = self.local_array.dtype
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size * self.itemsize


    def __del__(self):
        # If the __init__ method fails, we may not have a valid comm
        # attribute and this needs to be protected against.
        if hasattr(self, 'comm'):
            if self.comm is not None:
                try:
                    self.comm.Free()
                except:
                    pass

    def compatibility_hash(self):
        return hash((self.shape, self.dist, self.grid_shape, True))


def arecompatible(a, b):
    """Do these arrays have the same compatibility hash?"""
    return a.compatibility_hash() == b.compatibility_hash()