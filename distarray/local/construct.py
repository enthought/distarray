# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from __future__ import division

from itertools import product
from functools import reduce

import numpy

from distarray.mpiutils import MPI
from distarray.local.error import (DistError, InvalidGridShapeError,
                                   GridShapeError, NullCommError,
                                   InvalidBaseCommError)
from distarray import utils, mpiutils
from distarray.externals.six import next


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


def init_comm(base_comm, grid_shape):
    """Create an MPI communicator with a cartesian topology."""
    return base_comm.Create_cart(grid_shape, len(grid_shape) * (False,),
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

def init_grid_shape(dim_data, comm_size):
    """ Generate a `grid_shape` from dim_data.

    `dim_data` may not have `proc_grid_size` set for each dimension.
    
    """
    shape = tuple(dd['size'] for dd in dim_data)
    dist = tuple(dd['dist_type'] for dd in dim_data)
    distdims = tuple(i for (i, v) in enumerate(dist) if v != 'n')
    grid_shape = optimize_grid_shape(shape, distdims, comm_size)
    return validate_grid_shape(grid_shape, dim_data, comm_size)

def validate_grid_shape(grid_shape, dim_data, comm_size):
    ''' Extracts the grid_shape from dim_data and validates it.'''
    # check that if dim_data[i]['dist_type'] == 'n' then grid_shape[i] == 1
    for dim, (gs, dd) in enumerate(zip(grid_shape, dim_data)):
        if dd['dist_type'] == 'n' and gs != 1:
            msg = "dimension %d is not distributed but has a grid size of %d"
            raise InvalidGridShapeError(msg % (dim, gs))
    if reduce(int.__mul__, grid_shape) != comm_size:
        msg = "grid shape %r not compatible with comm size of %d."
        raise InvalidGridShapeError(msg % (grid_shape, comm_size))
    return grid_shape


def optimize_grid_shape(shape, distdims, comm_size):
    ''' Attempts to allocate processes optimally for distributed dimensions.

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
    
    '''
    ndistdim = len(distdims)

    if ndistdim == 1:
        # Trivial case: all processes used for the one distributed dimension.
        dist_grid_shape = (comm_size,)

    elif comm_size == 1:
        # Trivial case: only one process to distribute over!
        dist_grid_shape = (1,) * ndistdim

    else: # Main case: comm_size > 1, ndistdim > 1.
        factors = utils.mult_partitions(comm_size, ndistdim)
        if not factors: # Can't factorize appropriately.
            raise GridShapeError("Cannot distribute array over processors.")

        reduced_shape = [shape[i] for i in distdims]

        # Reorder factors so they match the relative ordering in reduced_shape
        factors = [utils.mirror_sort(f, reduced_shape) for f in factors]

        # Pick the "best" factoring from `factors` according to which matches the
        # ratios among the dimensions in `shape`
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

    return grid_shape


def _compute_grid_ratios(shape):
    n = len(shape)
    ratios = []
    for (i, j) in product(range(n), range(n)):
        if i < j:
            ratios.append(shape[i] / shape[j])

    return numpy.array(ratios)


