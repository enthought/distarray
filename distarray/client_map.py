# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from itertools import product
from distarray.local.base import _start_stop_block

import numpy as np

def client_map_factory(size, dist, grid_size):

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
        raise NotImplementedError()

class ClientMDMap(object):
    '''
    Governs the mapping between global indices and process ranks for
    MultiDimensional objects.

    Works with the LocalMap classes to facilitate communication between global
    and local processes.

    '''

    def __init__(self, shape, dist, grid_shape):
        self.ndim = len(shape)
        self.shape = shape
        self.grid_shape = tuple(grid_shape) + (1,) * (len(shape) - len(grid_shape))
        if isinstance(dist, (list, tuple)):
            self.dist = dist + ['n'] * (self.ndim - len(dist))
        elif isinstance(dist, dict):
            self.dist = ['n'] * self.ndim
            for i, d in dist.items():
                self.dist[i] = d

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
