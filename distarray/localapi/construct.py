# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import division

from distarray.localapi.mpiutils import MPI
from distarray.localapi.error import NullCommError, InvalidBaseCommError


# ---------------------------------------------------------------------------
# Stateless functions for initializing various aspects of DistArray and
# LocalArray objects.
# ---------------------------------------------------------------------------

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
    elif isinstance(comm, MPI.Comm):
        return comm
    else:
        raise InvalidBaseCommError("Not an MPI.Comm instance")


def init_comm(base_comm, grid_shape):
    """Create an MPI communicator with a cartesian topology."""
    return base_comm.Create_cart(grid_shape, len(grid_shape) * (False,),
                                 reorder=False)
