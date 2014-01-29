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
import operator

from functools import reduce
from six import next

from distarray.local import construct


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
        self.dimdata = dimdata
        self.base_comm = construct.init_base_comm(comm)

        self.grid_shape = construct.init_grid_shape(self.shape,
                                                    self.distdims,
                                                    self.comm_size,
                                                    self.grid_shape)

        self.comm = construct.init_comm(self.base_comm, self.grid_shape,
                                        self.ndistdim)

        self.local_shape, self.maps = \
            construct.init_local_shape_and_maps(self.shape,
                                                self.grid_shape,
                                                self.distdims,
                                                self.map_classes)

        self.local_array = self._make_local_array(buf=buf, dtype=dtype)

        self.base = None
        self.ctypes = None

    @property
    def grid_shape(self):
        return tuple(dd.get('gridsize') for dd in self.dimdata
                     if dd.get('gridsize'))

    @grid_shape.setter
    def grid_shape(self, grid_shape):
        grid_size = iter(grid_shape)
        for dist, dd in zip(self.dist, self.dimdata):
            if dist:
                dd['gridsize'] = next(grid_size)

    @property
    def shape(self):
        return tuple(dd['datasize'] for dd in self.dimdata)

    @property
    def ndim(self):
        return len(self.dimdata)

    @property
    def size(self):
        return reduce(operator.mul, self.shape)

    @property
    def comm_size(self):
        return self.base_comm.Get_size()

    @property
    def comm_rank(self):
        return self.base_comm.Get_rank()

    @property
    def dist(self):
        return tuple(dd['disttype'] for dd in self.dimdata)

    @property
    def map_classes(self):
        return construct.init_map_classes(self.dist)

    @property
    def distdims(self):
        return tuple(i for (i, v) in enumerate(self.dist) if v)

    @property
    def ndistdim(self):
        return len(self.distdims)

    @property
    def cart_coords(self):
        rval = tuple(dd.get('gridrank') for dd in self.dimdata
                     if dd.get('gridrank'))
        assert rval == self.comm.Get_coords(self.comm_rank)
        return rval

    @property
    def local_size(self):
        return self.local_array.size

    @property
    def data(self):
        return self.local_array.data

    @property
    def dtype(self):
        return self.local_array.dtype

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

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
        return hash((self.shape, self.dist, self.grid_shape, True))

    def __del__(self):
        # If the __init__ method fails, we may not have a valid comm
        # attribute and this needs to be protected against.
        if hasattr(self, 'comm'):
            if self.comm is not None:
                try:
                    self.comm.Free()
                except:
                    pass


def arecompatible(a, b):
    """Do these arrays have the same compatibility hash?"""
    return a.compatibility_hash() == b.compatibility_hash()
