# encoding: utf-8
from __future__ import division

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

from itertools import product
from functools import reduce

import numpy

from distarray.mpiutils import MPI
from distarray.local.error import (DistError, InvalidGridShapeError,
                                   GridShapeError, NullCommError,
                                   InvalidBaseCommError)
from distarray import utils, mpiutils


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Stateless functions for initializing various aspects of LocalArray objects
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# These are functions rather than methods because they need to be both
# stateless and free of side-effects.  It is possible that they could be
# called multiple times and in multiple different contexts in the course
# of a LocalArray object's lifetime (for example upon a reshape or redist).
# The simplest and most robust way of insuring this is to get rid of 'self'
# (which holds all state) and make them standalone functions.

def init_base_comm(comm):
    """Sanitize an MPI.comm instance or create one."""
    if comm == MPI.COMM_NULL:
        raise NullCommError("Cannot create a LocalArray with COMM_NULL")
    elif comm is None:
        return mpiutils.COMM_PRIVATE
    elif isinstance(comm, MPI.Comm):
        return comm
    else:
        raise InvalidBaseCommError("Not an MPI.Comm instance")


def init_comm(base_comm, grid_shape, ndistdim):
    """Create an MPI communicator with a cartesian topology."""
    return base_comm.Create_cart(grid_shape, ndistdim * (False,),
                                 reorder=False)


def init_dist(dist, ndim):
    """Return a tuple containing dist-type for each dim.

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
    >>> init_dist({0: 'b', 3: 'c'}, 4)
    ('b', 'n', 'n', 'c')
    """
    if isinstance(dist, str):
        return ndim*(dist,)
    elif isinstance(dist, (list, tuple)):
        return tuple(dist)
    elif isinstance(dist, dict):
        return tuple([dist.get(i, 'n') for i in range(ndim)])
    else:
        DistError("Dist must be a string, tuple, list or dict")


def init_distdims(dist, ndim):
    """Return a tuple containing indices of distributed dimensions.

    Parameters
    ----------
    dist : tuple of str as returned from `init_dist`
    ndim : int

    Returns
    -------
    tuple of int

    Examples
    --------
    >>> init_distdims(('b', 'n', 'n', 'c'), 4)
    (0, 3)
    """
    reduced_dist = [d for d in dist if d != 'n']
    ndistdim = len(reduced_dist)
    if ndistdim > ndim:
        raise DistError("Too many distributed dimensions")
    distdims = [i for i in range(ndim) if dist[i] != 'n']
    return tuple(distdims)


def init_grid_shape(shape, distdims, comm_size, grid_shape=None):
    """Generate or validate a `grid_shape`.

    If `grid_shape` is None, generate a `grid_shape` using
    `optimize_grid_shape`.  Else, validate and sanitize the `grid_shape`
    given.
    """
    ndistdim = len(distdims)
    if grid_shape is None or grid_shape == tuple():
        grid_shape = optimize_grid_shape(shape, distdims, comm_size)
    else:
        try:
            grid_shape = tuple(grid_shape)
        except:
            msg = "grid_shape is not castable to a tuple."
            raise InvalidGridShapeError(msg)
    if len(grid_shape) != ndistdim:
        raise InvalidGridShapeError("grid_shape has the wrong length.")
    ngriddim = reduce(lambda x, y: x * y, grid_shape)
    if ngriddim != comm_size:
        msg = "grid_shape is incompatible with the number of processors."
        raise InvalidGridShapeError(msg)
    return tuple(int(s) for s in grid_shape)


def optimize_grid_shape(shape, distdims, comm_size):
    ndistdim = len(distdims)
    if ndistdim == 1:
        grid_shape = (comm_size,)
    else:
        factors = utils.mult_partitions(comm_size, ndistdim)
        if factors != []:
            reduced_shape = [shape[i] for i in distdims]
            factors = [utils.mirror_sort(f, reduced_shape) for f in factors]
            rs_ratio = _compute_grid_ratios(reduced_shape)
            f_ratios = [_compute_grid_ratios(f) for f in factors]
            distances = [rs_ratio-f_ratio for f_ratio in f_ratios]
            norms = numpy.array([numpy.linalg.norm(d, 2) for d in distances])
            index = norms.argmin()
            grid_shape = tuple(factors[index])
        else:
            raise GridShapeError("Cannot distribute array over processors.")
    return grid_shape


def _compute_grid_ratios(shape):
    n = len(shape)
    ratios = []
    for (i, j) in product(range(n), range(n)):
        if i < j:
            ratios.append(shape[i] / shape[j])

    return numpy.array(ratios)
