# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import operator
from itertools import product
from functools import reduce
from collections import Sequence, Mapping

import numpy

from distarray import utils
from distarray.externals.six import next, string_types
from distarray.externals.six.moves import map


class InvalidGridShapeError(Exception):
    pass


class GridShapeError(Exception):
    pass


def normalize_grid_shape(grid_shape, ndims):
    """ Adds 1's to grid_shape so it has `ndims` dimensions.
    """
    return tuple(grid_shape) + (1,) * (ndims - len(grid_shape))


def validate_grid_shape(grid_shape, dist, comm_size):
    """ Validates `grid_shape` tuple against the `dist` tuple and
    `comm_size`.
    """
    if len(grid_shape) != len(dist):
        msg = "grid_shape's length (%d) not equal to dist's length (%d)"
        raise InvalidGridShapeError(msg % (len(grid_shape), len(dist)))
    # check that if dist[i] == 'n' then grid_shape[i] == 1
    for dim, (gs, dd) in enumerate(zip(grid_shape, dist)):
        if dd == 'n' and gs != 1:
            msg = "dimension %s is not distributed but has a grid size of %s."
            raise InvalidGridShapeError(msg % (dim, gs))
    if reduce(operator.mul, grid_shape) != comm_size:
        msg = "grid shape %r not compatible with comm size of %d."
        raise InvalidGridShapeError(msg % (grid_shape, comm_size))
    return grid_shape


def make_grid_shape(shape, dist, comm_size):
    """ Generate a `grid_shape` from `shape` tuple and `dist` tuple.

    Does not assume that `dim_data` has `proc_grid_size` set for each
    dimension.

    Attempts to allocate processes optimally for distributed dimensions.

    Parameters
    ----------
        shape : tuple of int
            The global shape of the array.
        distdims : sequence of int
            The indices of the distributed dimensions.
        comm_size : int
            Total number of processes to distribute.

    Returns
    -------
        dist_grid_shape : tuple of int

    Raises
    ------
        GridShapeError if not possible to distribute `comm_size` processes over
        number of dimensions.
    """
    distdims = tuple(i for (i, v) in enumerate(dist) if v != 'n')
    ndistdim = len(distdims)

    if ndistdim == 1:
        # Trivial case: all processes used for the one distributed dimension.
        dist_grid_shape = (comm_size,)

    elif comm_size == 1:
        # Trivial case: only one process to distribute over!
        dist_grid_shape = (1,) * ndistdim

    else:  # Main case: comm_size > 1, ndistdim > 1.
        factors = utils.mult_partitions(comm_size, ndistdim)
        if not factors:  # Can't factorize appropriately.
            raise GridShapeError("Cannot distribute array over processors.")

        reduced_shape = [shape[i] for i in distdims]

        # Reorder factors so they match the relative ordering in reduced_shape
        factors = [utils.mirror_sort(f, reduced_shape) for f in factors]

        # Pick the "best" factoring from `factors` according to which matches
        # the ratios among the dimensions in `shape`.
        rs_ratio = _compute_grid_ratios(reduced_shape)
        f_ratios = [_compute_grid_ratios(f) for f in factors]
        distances = [rs_ratio-f_ratio for f_ratio in f_ratios]
        norms = numpy.array([numpy.linalg.norm(d, 2) for d in distances])
        index = norms.argmin()
        # we now have the grid shape for the distributed dimensions.
        dist_grid_shape = tuple(int(i) for i in factors[index])

    # Create the grid_shape, all 1's for now.
    grid_shape = [1] * len(shape)

    # Fill grid_shape in the distdim slots using dist_grid_shape
    it = iter(dist_grid_shape)
    for distdim in distdims:
        grid_shape[distdim] = next(it)

    return tuple(grid_shape)


def _compute_grid_ratios(shape):
    shape = tuple(map(float, shape))
    n = len(shape)
    ratios = []
    for (i, j) in product(range(n), range(n)):
        if i < j:
            ratios.append(shape[i] / shape[j])
    return numpy.array(ratios)


def normalize_dist(dist, ndim):
    """Return a tuple containing dist-type for each dimension.

    Parameters
    ----------
    dist : str, list, tuple, or dict
    ndim : int

    Returns
    -------
    tuple of str
        Contains string distribution type for each dim.

    Examples
    --------
    >>> normalize_dist({0: 'b', 3: 'c'}, 4)
    ('b', 'n', 'n', 'c')
    """
    if isinstance(dist, string_types):
        if len(dist) != 1:
            msg = "dist argument %r has must have length 1."
            raise ValueError(msg % (dist,))
        return ndim*(dist,)
    elif isinstance(dist, Sequence):
        return tuple(dist) + ('n',) * (ndim - len(dist))
    elif isinstance(dist, Mapping):
        return tuple(dist.get(i, 'n') for i in range(ndim))
    else:
        raise TypeError("Dist must be a string, tuple, list or dict")
