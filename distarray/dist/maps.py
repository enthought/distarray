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
from __future__ import absolute_import

import operator
from itertools import product
from abc import ABCMeta, abstractmethod

import numpy as np

from distarray.externals.six import add_metaclass
from distarray.externals.six.moves import range, reduce
from distarray.utils import remove_elements
from distarray.metadata_utils import (normalize_dist,
                                      normalize_grid_shape,
                                      make_grid_shape,
                                      positivify,
                                      _start_stop_block,
                                      normalize_dim_dict,
                                      normalize_reduction_axes)


def _dedup_dim_dicts(dim_dicts):
    """ Internal helper function to take a list of dimension dictionaries
    and remove the dupes.  What remains should be one dictionary per rank
    (for this dimension of the process grid).
    """
    # Workaround to make the dictionary's contents hashable.
    for d in dim_dicts:
        if 'indices' in d:
            d['indices'] = tuple(d['indices'])
    try:
        return [dict(u) for u in
                set(tuple(sorted(d.items())) for d in dim_dicts)]
    except TypeError:
        result = []
        for i, d in enumerate(dim_dicts):
            if d not in dim_dicts[i+1:]:
                result.append(d)
        return result


# ---------------------------------------------------------------------------
# Functions for creating Map objects
# ---------------------------------------------------------------------------

def choose_map(dist_type):
    """Choose a map class given one of the distribution types."""
    cls_from_dist_type = {
        'b': BlockMap,
        'c': BlockCyclicMap,
        'n': NoDistMap,
        'u': UnstructuredMap,
        }
    if dist_type not in cls_from_dist_type:
        raise ValueError("unknown distribution type for %r" % dist_type)
    return cls_from_dist_type[dist_type]


def _map_from_axis_dim_dicts(axis_dim_dicts):
    """ Generates a ClientMap instance from a sanitized sequence of
    dimension dictionaries.

    Parameters
    ----------
    axis_dim_dicts: sequence of dictionaries
        Each dictionary is a "dimension dictionary" from the distributed array
        protocol, one per process in this dimension of the process grid.  The
        dimension dictionaries shall all have the same keys and values for
        global attributes: `dist_type`, `size`, `proc_grid_size`, and perhaps
        others.

    Returns
    -------
        An instance of a subclass of MapBase.

    """
    # check that all processes / ranks are accounted for.
    proc_ranks = sorted(dd['proc_grid_rank'] for dd in axis_dim_dicts)
    if proc_ranks != list(range(len(axis_dim_dicts))):
        msg = "Ranks of processes (%r) not consistent."
        raise ValueError(msg % proc_ranks)
    # Sort axis_dim_dicts according to proc_grid_rank.
    axis_dim_dicts = sorted(axis_dim_dicts, key=lambda d: d['proc_grid_rank'])

    dist_type = axis_dim_dicts[0]['dist_type']
    map_class = choose_map(dist_type)
    return map_class.from_axis_dim_dicts(axis_dim_dicts)


def map_from_global_dim_dict(global_dim_dict):
    """Given a global_dim_dict return map."""

    dist_type = global_dim_dict['dist_type']
    map_class = choose_map(dist_type)
    return map_class.from_global_dim_dict(global_dim_dict)


def map_from_sizes(size, dist_type, grid_size):
    """ Returns an instance of the appropriate subclass of MapBase.
    """
    map_class = choose_map(dist_type)
    return map_class(size, grid_size)


# ---------------------------------------------------------------------------
# Map classes
# ---------------------------------------------------------------------------

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

    def is_compatible(self, map):
        return ((self.dist == map.dist) and
                (vars(self) == vars(map)))


# ---------------------------------------------------------------------------
# 1-D Map classes
# ---------------------------------------------------------------------------

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
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        if len(axis_dim_dicts) != 1:
            msg = ("Number of dimension dictionaries "
                   "non-unitary for non-distributed dimension.")
            raise ValueError(msg)
        dd = axis_dim_dicts[0]
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
        return [0] if 0 <= idx < self.size else []

    def get_dimdicts(self):
        return ({
            'dist_type': 'n',
            'size': self.size,
            'proc_grid_size': 1,
            'proc_grid_rank': 0,
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
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        self = cls.__new__(cls)
        dd = axis_dim_dicts[0]
        if dd['dist_type'] != 'b':
            msg = "Wrong dist_type (%r) for block map."
            raise ValueError(msg % dd['dist_type'])
        self.size = dd['size']
        self.grid_size = dd['proc_grid_size']
        if self.grid_size != len(axis_dim_dicts):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(axis_dim_dicts), self.grid_size))
        self.bounds = [(d['start'], d['stop']) for d in axis_dim_dicts]
        self.boundary_padding, self.comm_padding = dd.get('padding', (0, 0))

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
            if lower <= idx < upper:
                coords.append(coord)
        return coords

    def get_dimdicts(self):
        grid_ranks = range(len(self.bounds))
        cpadding = self.comm_padding
        padding = [[cpadding, cpadding] for i in grid_ranks]
        padding[0][0] = self.boundary_padding
        padding[-1][-1] = self.boundary_padding
        data_tuples = zip(grid_ranks, padding, self.bounds)
        # Build the result
        out = []
        for grid_rank, padding, (start, stop) in data_tuples:
            out.append({
                'dist_type': 'b',
                'size': self.size,
                'proc_grid_size': self.grid_size,
                'proc_grid_rank': grid_rank,
                'start': start,
                'stop': stop,
                'padding': padding,
                })
        return tuple(out)


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
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        dd = axis_dim_dicts[0]
        if dd['dist_type'] != 'c':
            msg = "Wrong dist_type (%r) for cyclic map."
            raise ValueError(msg % dd['dist_type'])
        size = dd['size']
        grid_size = dd['proc_grid_size']
        if grid_size != len(axis_dim_dicts):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(axis_dim_dicts), grid_size))
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
        return tuple(({'dist_type': 'c',
                        'size': self.size,
                        'proc_grid_size': self.grid_size,
                        'proc_grid_rank': grid_rank,
                        'start': grid_rank * self.block_size,
                        'block_size': self.block_size,
                        }) for grid_rank in range(self.grid_size))


class UnstructuredMap(MapBase):

    dist = 'u'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        if glb_dim_dict['dist_type'] != 'u':
            msg = "Wrong dist_type (%r) for unstructured map."
            raise ValueError(msg % glb_dim_dict['dist_type'])
        indices = tuple(np.asarray(i) for i in glb_dim_dict['indices'])
        size = sum(len(i) for i in indices)
        grid_size = len(indices)
        return cls(size, grid_size, indices=indices)

    @classmethod
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        dd = axis_dim_dicts[0]
        if dd['dist_type'] != 'u':
            msg = "Wrong dist_type (%r) for unstructured map."
            raise ValueError(msg % dd['dist_type'])
        size = dd['size']
        grid_size = dd['proc_grid_size']
        if grid_size != len(axis_dim_dicts):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(axis_dim_dicts), grid_size))
        indices = [dd['indices'] for dd in axis_dim_dicts]
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
        return tuple(({
            'dist_type': 'u',
            'size': self.size,
            'proc_grid_size': self.grid_size,
            'proc_grid_rank': grid_rank,
            'indices': ii,
            }) for grid_rank, ii in enumerate(self.indices))


# ---------------------------------------------------------------------------
# N-Dimensional map.
# ---------------------------------------------------------------------------

class Distribution(object):

    """ Governs the mapping between global indices and process ranks for
    multi-dimensional objects.
    """

    @classmethod
    def from_dim_data_per_rank(cls, context, dim_data_per_rank, targets=None):
        """ Create a Distribution from a sequence of `dim_data` tuples. """

        self = cls.__new__(cls)
        dd0 = dim_data_per_rank[0]
        self.context = context
        self.targets = sorted(targets or context.targets)
        self.comm = self.context._make_subcomm(self.targets)
        for dim_data in dim_data_per_rank:
            for dim_dict in dim_data:
                normalize_dim_dict(dim_dict)
        self.shape = tuple(dd['size'] for dd in dd0)
        self.ndim = len(dd0)
        self.dist = tuple(dd['dist_type'] for dd in dd0)
        self.grid_shape = tuple(dd['proc_grid_size'] for dd in dd0)
        self.grid_shape = normalize_grid_shape(self.grid_shape, self.ndim,
                                               self.dist, len(self.targets))

        coords = [tuple(d['proc_grid_rank'] for d in dd) for dd in
                  dim_data_per_rank]

        self.rank_from_coords = np.empty(self.grid_shape, dtype=np.int32)
        for (r, c) in enumerate(coords):
            self.rank_from_coords[c] = r

        # `axis_dim_dicts_per_axis` is the zip of `dim_data_per_rank`,
        # with duplicates removed.  It is a list of `axis_dim_dicts`.
        # Each `axis_dim_dicts` is a list of dimension dictionaries, one per
        # process on a single axis of the process grid.
        axis_dim_dicts_per_axis = [_dedup_dim_dicts(axis_dim_dicts)
                                   for axis_dim_dicts in
                                   zip(*dim_data_per_rank)]

        if len(axis_dim_dicts_per_axis) != self.ndim:
            raise ValueError("Inconsistent dimensions.")

        self.maps = [_map_from_axis_dim_dicts(axis_dim_dicts) for
                     axis_dim_dicts in axis_dim_dicts_per_axis]

        return self

    @classmethod
    def from_shape(cls, context, shape, dist=None, grid_shape=None,
                   targets=None):

        # special case when dist is all 'n's.
        if (dist is not None) and all(d == 'n' for d in dist):
            if (targets is not None) and (len(targets) != 1):
                raise ValueError('target dist conflict')
            elif targets is None:
                targets = [context.targets[0]]
            else:
                # then targets is set correctly
                pass

        self = cls.__new__(cls)
        self.context = context
        self.targets = sorted(targets or context.targets)
        self.comm = self.context._make_subcomm(self.targets)
        self.shape = shape
        self.ndim = len(shape)

        # dist
        if dist is None:
            dist = {0: 'b'}
        self.dist = normalize_dist(dist, self.ndim)

        # grid_shape
        if grid_shape is None:
            grid_shape = make_grid_shape(self.shape, self.dist,
                                         len(self.targets))

        self.grid_shape = normalize_grid_shape(grid_shape, self.ndim,
                                               self.dist, len(self.targets))

        # TODO: FIXME: assert that self.rank_from_coords is valid and conforms
        # to how MPI does it.
        nelts = reduce(operator.mul, self.grid_shape, 1)
        self.rank_from_coords = np.arange(nelts).reshape(self.grid_shape)

        # List of `ClientMap` objects, one per dimension.
        self.maps = [map_from_sizes(*args)
                     for args in zip(self.shape, self.dist, self.grid_shape)]
        return self

    def __init__(self, context, global_dim_data, targets=None):
        """Make a Distribution from a global_dim_data structure.

        Parameters
        ----------
        global_dim_data : tuple of dict
            A global dimension dictionary per dimension.  See following `Note`
            section.

        Returns
        -------
        result : Distribution
            An empty DistArray of the specified size, dimensionality, and
            distribution.

        Note
        ----

        The `global_dim_data` tuple is a simple, straightforward data structure
        that allows full control over all aspects of a DistArray's distribution
        information.  It does not contain any of the array's *data*, only the
        *metadata* needed to specify how the array is to be distributed.  Each
        dimension of the array is represented by corresponding dictionary in
        the tuple, one per dimension.  All dictionaries have a `dist_type` key
        that specifies whether the array is block, cyclic, or unstructured.
        The other keys in the dictionary are dependent on the `dist_type` key.

        **Block**

        * ``dist_type`` is ``'b'``.

        * ``bounds`` is a sequence of integers, at least two elements.

          The ``bounds`` sequence always starts with 0 and ends with the global
          ``size`` of the array.  The other elements indicate the local array
          global index boundaries, such that successive pairs of elements from
          ``bounds`` indicates the ``start`` and ``stop`` indices of the
          corresponding local array.

        * ``comm_padding`` integer, greater than or equal to zero.
        * ``boundary_padding`` integer, greater than or equal to zero.

        These integer values indicate the communication or boundary padding,
        respectively, for the local arrays.  Currently only a single value for
        both ``boundary_padding`` and ``comm_padding`` is allowed for the
        entire dimension.

        **Cyclic**

        * ``dist_type`` is ``'c'``

        * ``proc_grid_size`` integer, greater than or equal to one.

        The size of the process grid in this dimension.  Equivalent to the
        number of local arrays in this dimension and determines the number of
        array sections.

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.

        * ``block_size`` integer, optional.  Greater than or equal to one.

        If not present, equivalent to being present with value of one.

        **Unstructured**

        * ``dist_type`` is ``'u'``

        * ``indices`` sequence of one-dimensional numpy integer arrays or
          buffers.

          The ``len(indices)`` is the number of local unstructured arrays in
          this dimension.

          To compute the global size of the array in this dimension, compute
          ``sum(len(ii) for ii in indices)``.

        **Not-distributed**

        The ``'n'`` distribution type is a convenience to specify that an array
        is not distributed along this dimension.

        * ``dist_type`` is ``'n'``

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.
        """
        self.context = context
        self.targets = sorted(targets or context.targets)
        self.comm = self.context._make_subcomm(self.targets)
        self.maps = [map_from_global_dim_dict(gdd) for gdd in global_dim_data]
        self.shape = tuple(m.size for m in self.maps)
        self.ndim = len(self.maps)
        self.dist = tuple(m.dist for m in self.maps)
        self.grid_shape = tuple(m.grid_size for m in self.maps)

        self.grid_shape = normalize_grid_shape(self.grid_shape, self.ndim,
                                               self.dist, len(self.targets))

        nelts = reduce(operator.mul, self.grid_shape, 1)
        self.rank_from_coords = np.arange(nelts).reshape(self.grid_shape)

    @property
    def has_precise_index(self):
        """
        Does the client-side Distribution know precisely who owns all indices?

        This can be used to determine whether one needs to use the `checked`
        version of `__getitem__` or `__setitem__` on LocalArrays.
        """
        return not any(isinstance(m, UnstructuredMap) for m in self.maps)

    def owning_ranks(self, idxs):
        """ Returns a list of ranks that may *possibly* own the location in the
        `idxs` tuple.

        For many distribution types, the owning rank is precisely known; for
        others, it is only probably known.  When the rank is precisely known,
        `owning_ranks()` returns a list of exactly one rank.  Otherwise,
        returns a list of more than one rank.

        If the `idxs` tuple is out of bounds, raises `IndexError`.
        """
        idxs = map(positivify, idxs, self.shape) # positivify and check
        dim_coord_hits = [m.owners(idx) for (m, idx) in zip(self.maps, idxs)]
        all_coords = product(*dim_coord_hits)
        ranks = [self.rank_from_coords[c] for c in all_coords]
        return ranks

    def owning_targets(self, idxs):
        """ Like `owning_ranks()` but returns a list of targets rather than
        ranks.

        Convenience method meant for IPython parallel usage.
        """
        return [self.targets[r] for r in self.owning_ranks(idxs)]

    def get_dim_data_per_rank(self):
        dds = [enumerate(m.get_dimdicts()) for m in self.maps]
        if not dds:
            return []
        cart_dds = product(*dds)
        coord_and_dd = [zip(*cdd) for cdd in cart_dds]
        rank_and_dd = sorted((self.rank_from_coords[c], dd) for (c, dd) in coord_and_dd)
        return [dd for (_, dd) in rank_and_dd]

    def is_compatible(self, o):
        return ((self.context, self.targets, self.shape, self.ndim, self.dist, self.grid_shape) ==
                (o.context,    o.targets,    o.shape,    o.ndim,    o.dist,    o.grid_shape) and
                all(m.is_compatible(om) for (m, om) in zip(self.maps, o.maps)))

    def reduce(self, axes):
        """
        Returns a new Distribution reduced along `axis`, i.e., the new
        distribution has one fewer dimension than `self`.
        """

        # the `axis` argument can actually be a sequence of axes, so we rename it.
        axes = normalize_reduction_axes(axes, self.ndim)

        reduced_shape = remove_elements(axes, self.shape)
        reduced_dist = remove_elements(axes, self.dist)
        reduced_grid_shape = remove_elements(axes, self.grid_shape)

        # This block is required because np.min() works one axis at a time.
        reduced_ranks = self.rank_from_coords.copy()
        for axis in axes:
            reduced_ranks = np.min(reduced_ranks, axis=axis, keepdims=True)

        reduced_targets = [self.targets[r] for r in reduced_ranks.flat]

        return Distribution.from_shape(context=self.context,
                                       shape=reduced_shape,
                                       dist=reduced_dist,
                                       grid_shape=reduced_grid_shape,
                                       targets=reduced_targets)
