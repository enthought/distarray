# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
ClientMDMap class and auxiliary ClientMap classes.

The ClientMDMap is a multi-dimensional map class that manages the
one-dimensional maps for each DistArray dimension.  The ClientMDMap class
represents the *distribution* information for a distributed array, independent
of the distributed array's *data*. ClientMDMaps allow DistArrays to reduce
overall communication when indexing and slicing by determining which processes
own (or may possibly own) the indices in question.  Two DistArray objects can
share the same ClientMDMap if they have the exact same distribution.

The one-dimensional ClientMap classes keep track of which process owns which
index in that dimension.  This class has several subclasses for specific
distribution types, including `BlockMap`, `CyclicMap`, `NoDistMap`, and
`UnstructuredMap`.

"""

import operator
from itertools import product
from abc import ABCMeta, abstractmethod

import numpy as np

from distarray.externals.six import add_metaclass
from distarray.externals.six.moves import range, reduce
from distarray.local.localarray import _start_stop_block
from distarray.metadata_utils import (normalize_dist,
                                      normalize_grid_shape,
                                      make_grid_shape,
                                      validate_grid_shape)


def client_map_factory(size, dist, grid_size):
    """ Returns an instance of the appropriate subclass of ClientMapBase.
    """
    cls_from_dist = {
            'b': ClientBlockMap,
            'c': ClientCyclicMap,
            'n': ClientNoDistMap,
            'u': ClientUnstructuredMap,
            }
    if dist not in cls_from_dist:
        raise ValueError("unknown distribution type for %r" % dist)
    return cls_from_dist[dist](size, grid_size)


@add_metaclass(ABCMeta)
class ClientMapBase(object):
    """ Base class for one-dimensional client-side maps.

    Maps keep track of the relevant distribution information for a single
    dimension of a distributed array.  Maps allow distributed arrays to keep
    track of which process to talk to when indexing and slicing.

    Classes that inherit from `ClientMapBase` must implement the `owners()`
    abstractmethod.

    """

    @abstractmethod
    def owners(self, idx):
        """ Returns a list of process IDs in this dimension that might possibly
        own `idx`.

        Raises `IndexError` if `idx` is out of bounds.

        """
        raise IndexError()


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

    def __init__(self, context, shape, dist, grid_shape=None):

        self.context = context
        self.shape = shape
        self.ndim = len(shape)
        self.dist = normalize_dist(dist, self.ndim)

        if grid_shape is None:  # Make a new grid_shape if not provided.
            self.grid_shape = make_grid_shape(self.shape, dist,
                                              len(context.targets))
        else:  # Otherwise normalize the one passed in.
            self.grid_shape = normalize_grid_shape(grid_shape, self.ndim)
        # In either case, validate.
        validate_grid_shape(self.grid_shape, self.dist, len(context.targets))

        # TODO: FIXME: assert that self.rank_from_coords is valid and conforms
        # to how MPI does it.
        nelts = reduce(operator.mul, self.grid_shape)
        self.rank_from_coords = np.arange(nelts).reshape(*self.grid_shape)

        # List of `ClientMap` objects, one per dimension.
        self.maps = [client_map_factory(*args)
                     for args in zip(self.shape, self.dist, self.grid_shape)]

    def owning_ranks(self, idxs):
        """ Returns a list of ranks that may *possibly* own the location in the
        `idxs` tuple.

        For many distribution types, the owning rank is precisely known; for
        others, it is only probably known.  When the rank is precisely known,
        `owning_ranks()` returns a list of exactly one rank.  Otherwise,
        returns a list of more than one rank.

        If the `idxs` tuple is out of bounds, raises `IndexError`.

        """
        dim_coord_hits = [m.owners(idx) for (m, idx) in zip(self.maps, idxs)]
        all_coords = product(*dim_coord_hits)
        ranks = [self.rank_from_coords[c] for c in all_coords]
        return ranks

    def owning_targets(self, idxs):
        """ Like `owning_ranks()` but returns a list of targets rather than
        ranks.

        Convenience method meant for IPython parallel usage.

        """
        return [self.context.targets[r] for r in self.owning_ranks(idxs)]
