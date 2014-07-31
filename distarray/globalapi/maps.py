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

from __future__ import division, absolute_import

import operator
from itertools import product
from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np

from distarray.externals.six import add_metaclass
from distarray.externals.six.moves import range, reduce
from distarray.utils import remove_elements
from distarray.metadata_utils import (normalize_dist,
                                      normalize_grid_shape,
                                      normalize_dim_dict,
                                      normalize_reduction_axes,
                                      make_grid_shape,
                                      sanitize_indices,
                                      _start_stop_block,
                                      tuple_intersection,
                                      shapes_from_dim_data_per_rank)


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

    """Base class for one-dimensional client-side maps.

    Maps keep track of the relevant distribution information for a single
    dimension of a distributed array.  Maps allow distributed arrays to keep
    track of which process to talk to when indexing and slicing.

    Classes that inherit from `MapBase` must implement the `index_owners()`
    abstractmethod.

    """

    @classmethod
    @abstractmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        """Make a Map from a global dimension dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        """Make a Map from a sequence of process-local dimension dictionaries.

        There should be one such dimension dictionary per process.
        """
        pass

    @abstractmethod
    def __init__(self):
        """Create a new Map.  Parameters may vary for different subtypes."""
        pass

    @abstractmethod
    def index_owners(self, idx):
        """ Returns a list of process IDs in this dimension that might possibly
        own `idx`.

        Raises `IndexError` if `idx` is out of bounds.

        """
        raise IndexError()

    @abstractmethod
    def get_dimdicts(self):
        """Return a dim_dict per process in this dimension."""
        pass

    def _is_compatible_degenerate(self, map):
        right_types = all(isinstance(m, (NoDistMap, BlockMap, BlockCyclicMap))
                          for m in (self, map))
        return (right_types
                and self.grid_size == map.grid_size == 1
                and self.size == map.size)

    def is_compatible(self, map):
        if self._is_compatible_degenerate(map):
            return True
        else:
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
        self.grid_size = grid_size

    def index_owners(self, idx):
        return [0] if 0 <= idx < self.size else []

    def slice_owners(self, idx):
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else self.size
        step = idx.step if idx.step is not None else 1
        if tuple_intersection((start, stop, step), (0, self.size)):
            return [0]
        else:
            return []

    def get_dimdicts(self):
        return ({
            'dist_type': 'n',
            'size': self.size,
            'proc_grid_size': 1,
            'proc_grid_rank': 0,
            },)

    def slice(self, idx):
        """Make a new Map from a slice."""
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else self.size
        step = idx.step if idx.step is not None else 1
        isection = tuple_intersection((start, stop, step), (0, self.size))
        if isection:
            step = idx.step if idx.step is not None else 1
            isection_size = int(np.ceil((isection[1] - isection[0]) / step))
        else:
            isection_size = 0
        return self.__class__(size=isection_size, grid_size=1)

    def view(self, new_dimsize):
        """Scale this map for the `view` method."""
        return self.__class__(size=int(new_dimsize), grid_size=1)

    def is_compatible(self, other):
        return (isinstance(other, (NoDistMap, BlockMap, BlockCyclicMap)) and
                other.grid_size == self.grid_size and 
                other.size == self.size)


class BlockMap(MapBase):

    dist = 'b'

    @classmethod
    def from_global_dim_dict(cls, glb_dim_dict):
        if glb_dim_dict['dist_type'] != 'b':
            msg = "Wrong dist_type (%r) for block map."
            raise ValueError(msg % glb_dim_dict['dist_type'])

        bounds = glb_dim_dict['bounds']
        tuple_bounds = list(zip(bounds[:-1], bounds[1:]))

        size = bounds[-1]
        grid_size = max(len(bounds) - 1, 1)

        comm_padding = int(glb_dim_dict.get('comm_padding', 0))
        boundary_padding = int(glb_dim_dict.get('boundary_padding', 0))

        return cls(size=size, grid_size=grid_size, bounds=tuple_bounds,
                   comm_padding=comm_padding,
                   boundary_padding=boundary_padding)

    @classmethod
    def from_axis_dim_dicts(cls, axis_dim_dicts):
        dd = axis_dim_dicts[0]
        if dd['dist_type'] != 'b':
            msg = "Wrong dist_type (%r) for block map."
            raise ValueError(msg % dd['dist_type'])

        size = dd['size']
        grid_size = dd['proc_grid_size']
        if grid_size != len(axis_dim_dicts):
            msg = ("Number of dimension dictionaries (%r)"
                   "inconsistent with proc_grid_size (%r).")
            raise ValueError(msg % (len(axis_dim_dicts), grid_size))
        bounds = [(d['start'], d['stop']) for d in axis_dim_dicts]
        boundary_padding, comm_padding = dd.get('padding', (0, 0))

        return cls(size=size, grid_size=grid_size, bounds=bounds,
                   comm_padding=comm_padding,
                   boundary_padding=boundary_padding)

    def __init__(self, size, grid_size, bounds=None,
                 comm_padding=None, boundary_padding=None):
        self.size = size
        self.grid_size = grid_size
        if bounds is None:
            self.bounds = [_start_stop_block(size, grid_size, grid_rank)
                           for grid_rank in range(grid_size)]
        else:
            self.bounds = bounds
        self.comm_padding = comm_padding or 0
        self.boundary_padding = boundary_padding or 0

    def index_owners(self, idx):
        coords = []
        for (coord, (lower, upper)) in enumerate(self.bounds):
            if lower <= idx < upper:
                coords.append(coord)
        return coords

    def slice_owners(self, idx):
        coords = []
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else self.size
        step = idx.step if idx.step is not None else 1
        for (coord, (lower, upper)) in enumerate(self.bounds):
            if tuple_intersection((start, stop, step), (lower, upper)):
                coords.append(coord)
        return coords

    def get_dimdicts(self):
        bounds = self.bounds or [[0, 0]]
        grid_ranks = range(len(bounds))
        cpadding = self.comm_padding
        padding = [[cpadding, cpadding] for _ in grid_ranks]
        if len(padding) > 0:
            padding[0][0] = self.boundary_padding
            padding[-1][-1] = self.boundary_padding
        data_tuples = zip(grid_ranks, padding, bounds)
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

    def slice(self, idx):
        """Make a new Map from a slice."""
        new_bounds = [0]
        start = idx.start if idx.start is not None else 0
        step = idx.step if idx.step is not None else 1
        # iterate over the processes in this dimension
        for proc_start, proc_stop in self.bounds:
            stop = idx.stop if idx.stop is not None else proc_stop
            isection = tuple_intersection((start, stop, step),
                                          (proc_start, proc_stop))
            if isection:
                isection_size = int(np.ceil((isection[1] - (isection[0])) / step))
                new_bounds.append(isection_size + new_bounds[-1])
        if new_bounds == [0]:
            new_bounds = []

        size = new_bounds[-1] if len(new_bounds) > 0 else 0
        grid_size = max(len(new_bounds) - 1, 1)
        new_bounds = list(zip(new_bounds[:-1], new_bounds[1:]))
        return self.__class__(size=size, grid_size=grid_size,
                              bounds=new_bounds)

    def view(self, new_dimsize):
        """Scale this map for the `view` method."""
        factor = new_dimsize / self.size
        new_bounds = [(int(start*factor), int(stop*factor))
                      for (start, stop) in self.bounds]
        return self.__class__(size=int(new_dimsize), grid_size=self.grid_size,
                              bounds=new_bounds)

    def is_compatible(self, other):
        if isinstance(other, NoDistMap):
            return other.is_compatible(self)
        return super(BlockMap, self).is_compatible(other)


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

    def index_owners(self, idx):
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

    def is_compatible(self, other):
        if isinstance(other, NoDistMap):
            return other.is_compatible(self)
        return super(BlockCyclicMap, self).is_compatible(other)


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
        self._index_owners = range(self.grid_size)

    def index_owners(self, idx):
        # TODO: FIXME: for now, the unstructured map just returns all
        # processes.  Can be optimized if we know the upper and lower bounds
        # for each local array's global indices.
        return self._index_owners

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

def asdistribution(context, shape_or_dist, dist=None, grid_shape=None, targets=None):
    if isinstance(shape_or_dist, Distribution):
        return shape_or_dist
    return Distribution(context=context, shape=shape_or_dist,
                        dist=dist, grid_shape=grid_shape,
                        targets=targets)

class Distribution(object):

    """ Governs the mapping between global indices and process ranks for
    multi-dimensional objects.
    """

    @classmethod
    def from_maps(cls, context, maps, targets=None):
        """Create a Distribution from a sequence of `Map`\s.

        Parameters
        ----------
        context : Context object
        maps : Sequence of Map objects
        targets : Sequence of int, optional
            Sequence of engine target numbers. Default: all available

        Returns
        -------
        Distribution
        """
        # This constructor is called by all the others
        self = super(Distribution, cls).__new__(cls)
        self.context = context
        self.targets = sorted(targets or context.targets)
        self.comm = self.context.make_subcomm(self.targets)
        self.maps = maps
        self.shape = tuple(m.size for m in self.maps)
        self.ndim = len(self.maps)
        self.dist = tuple(m.dist for m in self.maps)
        self.grid_shape = tuple(m.grid_size for m in self.maps)

        self.grid_shape = normalize_grid_shape(self.grid_shape, self.shape,
                                               self.dist, len(self.targets))

        nelts = reduce(operator.mul, self.grid_shape, 1)
        self.rank_from_coords = np.arange(nelts).reshape(self.grid_shape)
        return self

    @classmethod
    def from_dim_data_per_rank(cls, context, dim_data_per_rank, targets=None):
        """Create a Distribution from a sequence of `dim_data` tuples.

        Parameters
        ----------
        context : Context object
        dim_data_per_rank : Sequence of dim_data tuples, one per rank
            See the "Distributed Array Protocol" for a description of
            dim_data tuples.
        targets : Sequence of int, optional
            Sequence of engine target numbers. Default: all available

        Returns
        -------
        Distribution
        """
        for dim_data in dim_data_per_rank:
            for dim_dict in dim_data:
                normalize_dim_dict(dim_dict)

        # `axis_dim_dicts_per_axis` is the zip of `dim_data_per_rank`,
        # with duplicates removed.  It is a list of `axis_dim_dicts`.
        # Each `axis_dim_dicts` is a list of dimension dictionaries, one per
        # process on a single axis of the process grid.
        axis_dim_dicts_per_axis = [_dedup_dim_dicts(axis_dim_dicts)
                                   for axis_dim_dicts in
                                   zip(*dim_data_per_rank)]

        ndim = len(dim_data_per_rank[0])
        if len(axis_dim_dicts_per_axis) != ndim:
            raise ValueError("Inconsistent dimensions.")

        maps = [_map_from_axis_dim_dicts(axis_dim_dicts) for
                axis_dim_dicts in axis_dim_dicts_per_axis]
        return cls.from_maps(context=context, maps=maps, targets=targets)

    def __new__(cls, context, shape, dist=None, grid_shape=None, targets=None):
        """Create a Distribution from a `shape` and other optional args.

        Parameters
        ----------
        context : Context object
        shape : tuple of int
            Shape of the resulting Distribution, one integer per dimension.
        dist : str, list, tuple, or dict, optional
            Shorthand data structure representing the distribution type for
            every dimension.  Default: {0: 'b'}, with all other dimensions 'n'.
        grid_shape : tuple of int
        targets : Sequence of int, optional
            Sequence of engine target numbers. Default: all available

        Returns
        -------
        Distribution
        """
        # special case when dist is all 'n's.
        if (dist is not None) and all(d == 'n' for d in dist):
            if (targets is not None) and (len(targets) != 1):
                raise ValueError('target dist conflict')
            elif targets is None:
                targets = [context.targets[0]]
            else:
                # then targets is set correctly
                pass

        ndim = len(shape)
        dist = dist or {0: 'b'}
        dist = normalize_dist(dist, ndim)

        targets = sorted(targets or context.targets)
        grid_shape = grid_shape or make_grid_shape(shape, dist, len(targets))
        grid_shape = normalize_grid_shape(grid_shape, shape, dist, len(targets))

        # choose targets from grid_shape
        ntargets = reduce(operator.mul, grid_shape, 1)
        targets = targets[:ntargets]

        # list of `ClientMap` objects, one per dimension.
        maps = [map_from_sizes(*args) for args in zip(shape, dist, grid_shape)]

        self = cls.from_maps(context=context, maps=maps, targets=targets)

        # TODO: FIXME: this is a workaround.  The reason we slice here is to
        # return a distribution with no empty local shapes.  The `from_maps()`
        # classmethod should be fixed to ensure no empty local arrays are
        # created in the first place.  That will remove the need to slice the
        # distribution to remove empty localshapes.
        if all(d in ('n', 'b') for d in self.dist):
            self = self.slice((slice(None),)*self.ndim)
        return self

    @classmethod
    def from_global_dim_data(cls, context, global_dim_data, targets=None):
        """Make a Distribution from a global_dim_data structure.

        Parameters
        ----------
        context : Context object
        global_dim_data : tuple of dict
            A global dimension dictionary per dimension.  See following `Note`
            section.
        targets : Sequence of int, optional
            Sequence of engine target numbers. Default: all available

        Returns
        -------
        Distribution

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
        maps = [map_from_global_dim_dict(gdd) for gdd in global_dim_data]
        return cls.from_maps(context=context, maps=maps, targets=targets)


    def __getitem__(self, idx):
        return self.maps[idx]

    def __len__(self):
        return len(self.maps)

    @property
    def has_precise_index(self):
        """
        Does the client-side Distribution know precisely who owns all indices?

        This can be used to determine whether one needs to use the `checked`
        version of `__getitem__` or `__setitem__` on LocalArrays.
        """
        return not any(isinstance(m, UnstructuredMap) for m in self.maps)

    def slice(self, index_tuple):
        """Make a new Distribution from a slice."""
        new_targets = self.owning_targets(index_tuple) or [0]
        new_maps = []
        # iterate over the dimensions
        for map_, idx in zip(self.maps, index_tuple):
            if isinstance(idx, Integral):
                continue  # integral indexing returns reduced dimensionality
            elif isinstance(idx, slice):
                new_maps.append(map_.slice(idx))
            else:
                msg = "Index must be a sequence of Integrals and slices."
                raise TypeError(msg)

        return self.__class__.from_maps(context=self.context,
                                        maps=new_maps,
                                        targets=new_targets)

    def owning_ranks(self, idxs):
        """ Returns a list of ranks that may *possibly* own the location in the
        `idxs` tuple.

        For many distribution types, the owning rank is precisely known; for
        others, it is only probably known.  When the rank is precisely known,
        `owning_ranks()` returns a list of exactly one rank.  Otherwise,
        returns a list of more than one rank.

        If the `idxs` tuple is out of bounds, raises `IndexError`.
        """
        _, idxs = sanitize_indices(idxs, ndim=self.ndim, shape=self.shape)
        dim_coord_hits = []
        for m, idx in zip(self.maps, idxs):
            if isinstance(idx, Integral):
                owners = m.index_owners(idx)
            elif isinstance(idx, slice):
                owners = m.slice_owners(idx)
            dim_coord_hits.append(owners)

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
        rank_and_dd = sorted((self.rank_from_coords[c], dd)
                             for (c, dd) in coord_and_dd)
        return [dd for (_, dd) in rank_and_dd]

    def is_compatible(self, o):
        return ((self.context, self.targets, self.shape, self.ndim, self.grid_shape) ==
                (o.context,    o.targets,    o.shape,    o.ndim,    o.grid_shape) and
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

        return Distribution(context=self.context,
                            shape=reduced_shape,
                            dist=reduced_dist,
                            grid_shape=reduced_grid_shape,
                            targets=reduced_targets)

    def view(self, new_dimsize=None):
        """Generate a new Distribution for use with DistArray.view."""
        if new_dimsize is None:
            return self
        scaled_map = self.maps[-1].view(new_dimsize)
        new_maps = self.maps[:-1] + [scaled_map]
        return self.__class__.from_maps(context=self.context, maps=new_maps)

    def localshapes(self):
        return shapes_from_dim_data_per_rank(self.get_dim_data_per_rank())
