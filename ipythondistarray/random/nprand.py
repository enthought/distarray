from ipythondistarray.core import distarray
from ipythondistarray.core.construct import find_local_shape, init_base_comm
from ipythondistarray.core.error import *
from ipythondistarray.mpi import mpibase

import numpy as np

# def __init__(self, shape, dtype=float, dist={0:'b'} , grid_shape=None,
#              comm=None, buf=None, offset=0):


__all__ = [
    'beta',
    'normal',
    'rand',
    'randint',
    'randn']


def beta(a, b, size=None, dist={0:'b'}, grid_shape=None, comm=None):
    if size is None:
        return np.random.beta(a, b)
    else:
        base_comm = init_base_comm(comm)
        comm_size = base_comm.Get_size()
        local_shape = find_local_shape(size, dist=dist, grid_shape=grid_shape, comm_size=comm_size)
        local_result = np.random.beta(a, b, size=local_shape)
        return distarray.DistArray(size, local_result.dtype, dist, grid_shape, comm, buf=local_result)


def normal(loc=0.0, scale=1.0, size=None, dist={0:'b'}, grid_shape=None, comm=None):
    if size is None:
        return np.random.normal(loc, scale)
    else:
        base_comm = init_base_comm(comm)
        comm_size = base_comm.Get_size()
        local_shape = find_local_shape(size, dist=dist, grid_shape=grid_shape, comm_size=comm_size)
        local_result = np.random.normal(loc, scale, size=local_shape)
        return distarray.DistArray(size, local_result.dtype, dist, grid_shape, comm, buf=local_result)


def rand(size=None, dist={0:'b'}, grid_shape=None, comm=None):
    if size is None:
        return np.random.rand()
    else:
        base_comm = init_base_comm(comm)
        comm_size = base_comm.Get_size()
        local_shape = find_local_shape(size, dist=dist, grid_shape=grid_shape, comm_size=comm_size)
        local_result = np.random.rand(*local_shape)
        return distarray.DistArray(size, local_result.dtype, dist, grid_shape, comm, buf=local_result)


def randint(low, high=None, size=None, dist={0:'b'}, grid_shape=None, comm=None):
    if size is None:
        return np.random.randint(low, high)
    else:
        base_comm = init_base_comm(comm)
        comm_size = base_comm.Get_size()
        local_shape = find_local_shape(size, dist=dist, grid_shape=grid_shape, comm_size=comm_size)
        local_result = np.random.randint(low, high, size=local_shape)
        return distarray.DistArray(size, local_result.dtype, dist, grid_shape, comm, buf=local_result)


def randn(size=None, dist={0:'b'}, grid_shape=None, comm=None):
    if size is None:
        return np.random.randn()
    else:
        base_comm = init_base_comm(comm)
        comm_size = base_comm.Get_size()
        local_shape = find_local_shape(size, dist=dist, grid_shape=grid_shape, comm_size=comm_size)
        local_result = np.random.randn(*local_shape)
        return distarray.DistArray(size, local_result.dtype, dist, grid_shape, comm, buf=local_result)
