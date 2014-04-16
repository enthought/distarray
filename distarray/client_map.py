# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Distribution class and auxiliary ClientMap classes.

The Distribution is a multi-dimensional map class that manages the
one-dimensional maps for each DistArray dimension.  The Distribution class
represents the *distribution* information for a distributed array, independent
of the distributed array's *data*. Distributions allow DistArrays to reduce
overall communication when indexing and slicing by determining which processes
own (or may possibly own) the indices in question.  Two DistArray objects can
share the same Distribution if they have the exact same distribution.

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
    """ Returns an instance of the appropriate subclass of MapBase.
    """
    cls_from_dist = {
            'b': BlockMap,
            'c': BlockCyclicMap,
            'n': NoDistMap,
            'u': ClientUnstructuredMap,
            }
    if dist not in cls_from_dist:
        raise ValueError("unknown distribution type for %r" % dist)
    return cls_from_dist[dist](size, grid_size)


@add_metaclass(ABCMeta)
class MapBase(object):
    """ Base class for one-dimensional client-side maps.

    Maps keep track of the relevant distribution information for a single
    dimension of a distributed array.  Maps allow distributed arrays to keep
    track of which process to talk to when indexing and slicing.

    Classes that inherit from `MapBase` must implement the `owners()`
    abstractmethod.

    """

    @abstractmethod
    def owners(self, idx):
        """ Returns a list of process IDs in this dimension that might possibly
        own `idx`.

        Raises `IndexError` if `idx` is out of bounds.

        """
        raise IndexError()


class NoDistMap(MapBase):

    dist = 'n'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        if glb_dim_dict['dist_type'] != 'n':
            msg = "Wrong dist_type (%r) for non-distributed map."
            raise ValueError(msg % glb_dim_dict['dist_type'])
        size = glb_dim_dict['size']
        return cls(size, grid_size=1)

    @classmethod
    def from_dim_data(cls, dim_data_seq):
        if len(dim_data_seq) != 1:
            msg = ("Number of dimension dictionaries "
                   "non-unitary for non-distributed dimension.")
            raise ValueError(msg)
        dd = dim_data_seq[0]
        if dd['dist_type'] != 'n':
            msg = "Wrong dist_type (%r) for non-distributed map."
            raise ValueError(msg % dd['dist_type'])
        grid_size = dd['proc_grid_size']
        size = dd['size']
        return cls(size, grid_size)

    def __init__(self, size, grid_size):
        if grid_size != 1:
            msg = "grid_size for NoDistMap must be 1 (given %s)"
            raise ValueError(msg % grid_size)
        self.size = size

    def owners(self, idx):
        return [0] if idx >= 0 and idx < self.size else []

    def get_dimdicts(self):
        return ({'dist_type' : 'n',
                'size' : self.size,
                'proc_grid_size' : 1,
                'proc_grid_rank' : 0,
                },)


class BlockMap(MapBase):

    dist = 'b'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):

        self = cls.__new__(cls)
        if glb_dim_dict['dist_type'] != 'b':
            msg = "Wrong dist_type (%r) for block map."
            raise ValueError(msg % glb_dim_dict['dist_type'])

        bounds = glb_dim_dict['bounds']
        self.bounds = list(zip(bounds[:-1], bounds[1:]))

        self.size = bounds[-1]
        self.grid_size = len(bounds) - 1

        self.comm_padding = int(glb_dim_dict.get('comm_padding', 0))
        self.boundary_padding = int(glb_dim_dict.get('boundary_padding', 0))

        return self

    @classmethod
    def from_dim_data(cls, dim_data_seq):
        self = cls.__new__(cls)
        dd = dim_data_seq[0]
        if dd['dist_type'] != 'b':
            msg = "Wrong dist_type (%r) for block map."
            raise ValueError(msg % dd['dist_type'])
        self.size = dd['size']
        self.grid_size = dd['proc_grid_size']
        if self.grid_size != len(dim_data_seq):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(dim_data_seq), self.grid_size))
        self.bounds = [(d['start'], d['stop']) for d in dim_data_seq]
        self.boundary_padding, self.comm_padding = dd.get('padding', (0,0))
        
        return self

    def __init__(self, size, grid_size):
        self.size = size
        self.grid_size = grid_size
        self.bounds = [_start_stop_block(size, grid_size, grid_rank)
                       for grid_rank in range(grid_size)]
        self.boundary_padding = self.comm_padding = 0

    def owners(self, idx):
        coords = []
        for (coord, (lower, upper)) in enumerate(self.bounds):
            if idx >= lower and idx < upper:
                coords.append(coord)
        return coords

    def get_dimdicts(self):
        grid_ranks = range(len(self.bounds))
        cpadding = self.comm_padding
        padding = [[cpadding, cpadding] for i in grid_ranks]
        padding[0][0] = self.boundary_padding
        padding[-1][-1] = self.boundary_padding
        data_tuples = zip(grid_ranks, padding, self.bounds)
        return tuple(({'dist_type' : 'b',
                        'size' : self.size,
                        'proc_grid_size' : self.grid_size,
                        'proc_grid_rank' : grid_rank,
                        'start' : start,
                        'stop' : stop,
                        'padding': padding,
                        }) for grid_rank, padding, (start, stop) in data_tuples)


class BlockCyclicMap(MapBase):

    dist = 'c'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        if glb_dim_dict['dist_type'] != 'c':
            msg = "Wrong dist_type (%r) for cyclic map."
            raise ValueError(msg % glb_dim_dict['dist_type'])
        size = glb_dim_dict['size']
        grid_size = glb_dim_dict['proc_grid_size']
        block_size = glb_dim_dict.get('block_size', 1)
        return cls(size, grid_size, block_size)

    @classmethod
    def from_dim_data(cls, dim_data_seq):
        dd = dim_data_seq[0]
        if dd['dist_type'] != 'c':
            msg = "Wrong dist_type (%r) for cyclic map."
            raise ValueError(msg % dd['dist_type'])
        size = dd['size']
        grid_size = dd['proc_grid_size']
        if grid_size != len(dim_data_seq):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(dim_data_seq), grid_size))
        block_size = dd.get('block_size', 1)
        return cls(size, grid_size, block_size)

    def __init__(self, size, grid_size, block_size=1):
        self.size = size
        self.grid_size = grid_size
        self.block_size = block_size

    def owners(self, idx):
        idx_block = idx // self.block_size
        return [idx_block % self.grid_size]

    def get_dimdicts(self):
        return tuple(({'dist_type' : 'c',
                        'size' : self.size,
                        'proc_grid_size' : self.grid_size,
                        'proc_grid_rank' : grid_rank,
                        'start' : grid_rank * self.block_size,
                        'block_size': self.block_size,
                        }) for grid_rank in range(self.grid_size))


class ClientUnstructuredMap(MapBase):

    dist = 'u'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        if glb_dim_dict['dist_type'] != 'u':
            msg = "Wrong dist_type (%r) for unstructured map."
            raise ValueError(msg % glb_dim_dict['dist_type'])
        indices_sequence = tuple(np.asarray(ind) for ind in glb_dim_dict['indices'])
        size = sum(len(ii) for ii in indices_sequence)
        grid_size = len(indices_sequence)
        return cls(size, grid_size, indices=indices_sequence)

    @classmethod
    def from_dim_data(cls, dim_data_seq):
        dd = dim_data_seq[0]
        if dd['dist_type'] != 'u':
            msg = "Wrong dist_type (%r) for unstructured map."
            raise ValueError(msg % dd['dist_type'])
        size = dd['size']
        grid_size = dd['proc_grid_size']
        if grid_size != len(dim_data_seq):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(dim_data_seq), grid_size))
        indices = [dd['indices'] for dd in dim_data_seq]
        return cls(size, grid_size, indices=indices)

    def __init__(self, size, grid_size, indices=None):
        self.size = size
        self.grid_size = grid_size
        self.indices = indices
        if self.indices is not None:
            # Convert to NumPy arrays if not already.
            self.indices = [np.asarray(ind) for ind in self.indices]
        self._owners = range(self.grid_size)

    def owners(self, idx):
        # TODO: FIXME: for now, the unstructured map just returns all
        # processes.  Can be optimized if we know the upper and lower bounds
        # for each local array's global indices.
        return self._owners

    def get_dimdicts(self):
        if self.indices is None:
            raise ValueError()
        return tuple(({'dist_type' : 'u',
                        'size' : self.size,
                        'proc_grid_size' : self.grid_size,
                        'proc_grid_rank' : grid_rank,
                        'indices' : ii,
                        }) for grid_rank, ii in enumerate(self.indices))


def _compactify_dicts(dicts):
    """ Internal helper function to take a list of dimension dictionaries with
    duplicates and remove the dupes.

    """
    # Workaround to make the dictionary's contents hashable.
    for d in dicts:
        if 'indices' in d:
            d['indices'] = tuple(d['indices'])
    try:
        return [dict(u) for u in set(tuple(sorted(d.items())) for d in dicts)]
    except TypeError:
        result = []
        for i, d in enumerate(dicts):
            if d not in dicts[i+1:]:
                result.append(d)
        return result


def map_from_dim_datas(dim_datas):
    """ Generates a ClientMap instance from a santized sequence of dim_data
    dictionaries.

    Parameters
    ----------
    dim_datas : sequence of dictionaries
        Each dictionary is a "dimension dictionary" from the distributed array
        protocol, one per process in this dimension of the process grid.  The
        dimension dictionaries shall all have the same keys and values for
        global attributes: `dist_type`, `size`, `proc_grid_size`, and perhaps
        others.

    Returns
    -------
        An instance of a subclass of MapBase.

    """
    # check that all proccesses / ranks are accounted for.
    proc_ranks = sorted(dd['proc_grid_rank'] for dd in dim_datas)
    if proc_ranks != list(range(len(dim_datas))):
        msg = "Ranks of processes (%r) not consistent."
        raise ValueError(msg % proc_ranks)
    # Sort dim_datas according to proc_grid_rank.
    dim_datas = sorted(dim_datas, key=lambda d: d['proc_grid_rank'])

    dist_type = dim_datas[0]['dist_type']
    selector = {'n': NoDistMap.from_dim_data,
                'b': BlockMap.from_dim_data,
                'c': BlockCyclicMap.from_dim_data,
                'u': ClientUnstructuredMap.from_dim_data}
    if dist_type not in selector:
        raise ValueError("Unknown dist_type %r" % dist_type)
    return selector[dist_type](dim_datas)

def map_from_global_dim_dict(global_dim_dict):

    dist_type = global_dim_dict['dist_type']
    selector = {'n': NoDistMap.from_global_dim_dict,
                'b': BlockMap.from_global_dim_dict,
                'c': BlockCyclicMap.from_global_dim_dict,
                'u': ClientUnstructuredMap.from_global_dim_dict,
                }
    if dist_type not in selector:
        raise ValueError("Unknown dist_type %r" % dist_type)
    return selector[dist_type](global_dim_dict)


class Distribution(object):
    """ Governs the mapping between global indices and process ranks for
    multi-dimensional objects.

    """

    @classmethod
    def from_global_dim_data(cls, context, glb_dim_data):
        self = cls.__new__(cls)
        self.context = context
        self.maps = [map_from_global_dim_dict(gdd) for gdd in glb_dim_data]
        self.shape = tuple(m.size for m in self.maps)
        self.ndim = len(self.maps)
        self.dist = tuple(m.dist for m in self.maps)
        self.grid_shape = tuple(m.grid_size for m in self.maps)

        validate_grid_shape(self.grid_shape, self.dist, len(context.targets))

        nelts = reduce(operator.mul, self.grid_shape)
        self.rank_from_coords = np.arange(nelts).reshape(*self.grid_shape)

        return self

    @classmethod
    def from_dim_data(cls, context, dim_datas):
        """ Creates a Distribution from a sequence of `dim_data` dictionary
        tuples from each LocalArray.
        """

        self = cls.__new__(cls)
        dd0 = dim_datas[0]
        self.context = context
        self.shape = tuple(dd['size'] for dd in dd0)
        self.ndim = len(dd0)
        self.dist = tuple(dd['dist_type'] for dd in dd0)
        self.grid_shape = tuple(dd['proc_grid_size'] for dd in dd0)

        validate_grid_shape(self.grid_shape, self.dist, len(context.targets))

        coords = [tuple(d['proc_grid_rank'] for d in dd) for dd in dim_datas]
        self.rank_from_coords = { c: r for (r, c) in enumerate(coords)}

        dim_data_per_dim = [_compactify_dicts(dict_tuple)
                            for dict_tuple in zip(*dim_datas)]

        if len(dim_data_per_dim) != self.ndim:
            raise ValueError("Inconsistent dimensions.")

        self.maps = [map_from_dim_datas(ddpd) for ddpd in dim_data_per_dim]

        return self

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

    def get_local_dim_datas(self):
        dds = [enumerate(m.get_dimdicts()) for m in self.maps]
        cart_dds = product(*dds)
        coord_and_dd = [zip(*cdd) for cdd in cart_dds]
        rank_and_dd = sorted((self.rank_from_coords[c], dd) for (c, dd) in coord_and_dd)
        return [dd for (_, dd) in rank_and_dd]
