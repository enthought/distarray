# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from collections import Mapping, Sequence
from itertools import product
from distarray.local.localarray import _start_stop_block

from distarray.externals.six.moves import range, reduce

import numpy as np

def client_map_factory(size, dist, grid_size):
    """ Returns an instance of the appropriate subclass of ClientMapBase.
    """
    cls_from_dist = {
            'b' : ClientBlockMap,
            'c' : ClientCyclicMap,
            'n' : ClientNoDistMap,
            'u' : ClientUnstructuredMap,
            }
    if dist not in cls_from_dist:
        raise ValueError("unknown distribution type for %r" % dist)
    return cls_from_dist[dist](size, grid_size)


class ClientMapBase(object):
    """ Base class for client-side maps.

    Maps keep track of the relevant distribution information.  Maps allow
    distributed arrays to keep track of which process to talk to when indexing
    and slicing arrays.
    """
    pass

class ClientNoDistMap(ClientMapBase):

    def __init__(self, size, grid_size):
        if grid_size != 1:
            msg = "grid_size for ClientNoDistMap must be 1 (given %s)"
            raise ValueError(msg % grid_size)
        self.size = size

    def owners(self, idx):
        return [0] if idx >= 0 and idx < self.size else []

class ClientBlockMap(ClientMapBase):

    def __init__(self, size, grid_size):
        self.size = size
        self.grid_size = grid_size
        self.bounds = [_start_stop_block(size, grid_size, grid_rank)
                        for grid_rank in range(grid_size)]

    def owners(self, idx):
        coords = []
        for (coord, (lower, upper)) in enumerate(self.bounds):
            if idx >= lower and idx < upper:
                coords.append(coord)
        return coords

class ClientCyclicMap(ClientMapBase):
    
    def __init__(self, size, grid_size):
        self.size = size
        self.grid_size = grid_size

    def owners(self, idx):
        return [idx % self.grid_size]
        

class ClientUnstructuredMap(ClientMapBase):

    def __init__(self, size, grid_size):
        self.size = size
        self.grid_size = grid_size
        self._owners = range(self.grid_size)

    def owners(self, idx):
        # TODO: FIXME: for now, the unstructured map just returns all
        # processes.  Can be optimized if we know the upper and lower bounds
        # for each local array's global indices.
        return self._owners


class ClientMDMap(object):
    """ Governs the mapping between global indices and process ranks for
    multi-dimensional objects.
    """

    def __init__(self, shape, dist, grid_shape):
        self.ndim = len(shape)
        self.shape = shape
        self.grid_shape = tuple(grid_shape) + (1,) * (len(shape) - len(grid_shape))
        self.dist = _normalize_dist(self.ndim, dist)
        # TODO: FIXME: assert that self.rank_from_coords is valid and conforms
        # to how MPI does it.
        nelts = reduce(int.__mul__, grid_shape)
        self.rank_from_coords = np.arange(nelts).reshape(*self.grid_shape)

        self.maps = [client_map_factory(ss, dd, gg) 
                     for (ss, dd, gg) in zip(self.shape, self.dist, self.grid_shape)]


    def owning_ranks(self, idxs):
        dim_coord_hits = [m.owners(idx) for (m, idx) in zip(self.maps, idxs)]
        all_coords = product(*dim_coord_hits)
        ranks = [self.rank_from_coords[c] for c in all_coords]
        return ranks


def _normalize_dist(ndim, dist):
    """ If `dist` is a dictionary, convert it into the equivalent tuple.
    """
    if isinstance(dist, Sequence):
        return tuple(dist) + ('n',) * (ndim - len(dist))
    elif isinstance(dist, Mapping):
        dist_seq = ['n'] * ndim
        for i, d in dist.items():
            dist_seq[i] = d
        return tuple(dist_seq)
    else:
        raise TypeError("dist %r is not a Sequence or Mapping" % dist)
