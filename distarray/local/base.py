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
import operator

from functools import reduce
from distarray.externals.six import next

from distarray.local import construct, maps


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


def distribute_cyclic_indices(dd):
    """Fill in `start` given dimdict `dd`."""
    if 'start' in dd:
        return
    else:
        dd['start'] = dd['proc_grid_rank']


def distribute_indices(dim_data):
    """Fill in missing index related keys...

    for supported dist_types.
    """
    distribute_fn = {
        'b': distribute_block_indices,
        'c': distribute_cyclic_indices,
    }
    for dim in dim_data:
        if dim['dist_type'] != 'n':
            distribute_fn[dim['dist_type']](dim)


class BaseLocalArray(object):

    """Distributed memory Python arrays."""

    __array_priority__ = 20.0

    def __init__(self, dim_data, dtype=None, buf=None, comm=None):
        """Make a BaseLocalArray from a `dim_data` tuple.

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
        comm : MPI communicator object, optional

        Returns
        -------
        BaseLocalArray
            A BaseLocalArray encapsulating `buf`, or else an empty
            (uninitialized) BaseLocalArray.
        """
        self.dim_data = dim_data
        self.base_comm = construct.init_base_comm(comm)

        self.grid_shape = construct.init_grid_shape(self.shape,
                                                    self.distdims,
                                                    self.comm_size,
                                                    self.grid_shape)

        self.comm = construct.init_comm(self.base_comm, self.grid_shape,
                                        self.ndistdim)

        self._cache_proc_grid_rank()
        distribute_indices(self.dim_data)
        self.maps = tuple(maps.IndexMap.from_dimdict(dimdict) for dimdict in
                          dim_data)

        self.local_array = self._make_local_array(buf=buf, dtype=dtype)

        self.base = None
        self.ctypes = None

    @property
    def local_shape(self):
        return tuple(m.size for m in self.maps)

    @property
    def grid_shape(self):
        return tuple(dd.get('proc_grid_size') for dd in self.dim_data
                     if dd.get('proc_grid_size'))

    @grid_shape.setter
    def grid_shape(self, grid_shape):
        grid_size = iter(grid_shape)
        for dist, dd in zip(self.dist, self.dim_data):
            if dist != 'n':
                dd['proc_grid_size'] = next(grid_size)

    @property
    def shape(self):
        return tuple(dd['size'] for dd in self.dim_data)

    @property
    def ndim(self):
        return len(self.dim_data)

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
        return tuple(dd['dist_type'] for dd in self.dim_data)

    @property
    def distdims(self):
        return tuple(i for (i, v) in enumerate(self.dist) if v != 'n')

    @property
    def ndistdim(self):
        return len(self.distdims)

    @property
    def cart_coords(self):
        rval = tuple(dd.get('proc_grid_rank') for dd in self.dim_data
                     if dd.get('proc_grid_rank'))
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

    def _cache_proc_grid_rank(self):
        cart_coords = self.comm.Get_coords(self.comm_rank)
        dist_data = (self.dim_data[i] for i in self.distdims)
        for dim, cart_rank in zip(dist_data, cart_coords):
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
