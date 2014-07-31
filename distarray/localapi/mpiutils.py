# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Entry point for MPI.
"""

import numpy as np

from mpi4py import MPI
from distarray.error import InvalidCommSizeError, InvalidRankError


def get_comm_private():
    return MPI.COMM_WORLD.Clone()


def create_comm_of_size(size=4):
    """
    Create a subcommunicator of COMM_PRIVATE of given size.
    """
    COMM_PRIVATE = get_comm_private()
    group = COMM_PRIVATE.Get_group()
    comm_size = COMM_PRIVATE.Get_size()
    if size > comm_size:
        raise InvalidCommSizeError("requested size (%i) is bigger than the comm size (%i)" % (size, comm_size))
    else:
        subgroup = group.Incl(list(range(size)))
        newcomm = COMM_PRIVATE.Create(subgroup)
        return newcomm


def create_comm_with_list(nodes, base_comm=None):
    """
    Create a subcommunicator of base_comm with a list of ranks.

    If base_comm is not specified, defaults to COMM_PRIVATE.

    """
    base_comm = base_comm or get_comm_private()
    group = base_comm.Get_group()
    comm_size = base_comm.Get_size()
    size = len(nodes)
    if size > comm_size:
        raise InvalidCommSizeError("requested size (%i) is bigger than the comm size (%i)" % (size, comm_size))
    for i in nodes:
        if not i in range(comm_size):
            raise InvalidRankError("rank is not valid: %r" % i)
    subgroup = group.Incl(nodes)
    newcomm = base_comm.Create(subgroup)
    return newcomm


mpi_dtypes = {
    np.dtype('f'): MPI.FLOAT,
    np.dtype('d'): MPI.DOUBLE,
    np.dtype('i'): MPI.INTEGER,
    np.dtype('l'): MPI.LONG
}


def mpi_type_for_ndarray(a):
    return mpi_dtypes[a.dtype]
