#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------


import numpy as np

from ipythondistarray.mpi import mpibase
from ipythondistarray.mpi.mpibase import MPI
from ipythondistarray.core import maps
from ipythondistarray.core.error import *
from ipythondistarray import utils

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Stateless functions for initializing various aspects of DistArray objects
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# These are functions rather than methods because they need to be both
# stateless and free of side-effects.  It is possible that they could be
# called multiple times and in multiple different contexts in the course
# of a DistArray object's lifetime (for example upon a reshape or redist).
# The simplest and most robust way of insuring this is to get rid of 'self'
# (which holds all state) and make them standalone functions.


def init_base_comm(comm):
    if comm==MPI.COMM_NULL:
        raise MPICommError("Cannot create a DistArray with a MPI COMM_NULL")
    elif comm is None:
        return mpibase.COMM_PRIVATE
    elif isinstance(comm, MPI.Comm):
        return comm
    else:
        raise InvalidBaseCommError("Not an MPI.Comm instance")


def init_dist(dist, ndim):
    if isinstance(dist, str):
        return ndim*(dist,)
    elif isinstance(dist, (list, tuple)):
        return tuple(dist)
    elif isinstance(dist, dict):
        return tuple([dist.get(i) for i in range(ndim)])
    else:
        DistError("dist must be a string, tuple/list or dict") 


def init_distdims(dist, ndim):
    reduced_dist = [d for d in dist if d is not None]
    ndistdim = len(reduced_dist)
    if ndistdim > ndim:
        raise DistError("Too many distributed dimensions")
    distdims = [i for i in range(ndim) if dist[i] is not None]
    return tuple(distdims)


def init_map_classes(dist):
    reduced_dist = [d for d in dist if d is not None]
    map_classes = [maps.get_map_class(d) for d in reduced_dist]
    return tuple(map_classes)


def init_grid_shape(shape, grid_shape, distdims, comm_size):
    ndistdim = len(distdims)
    if grid_shape is None:
        grid_shape = optimize_grid_shape(shape, grid_shape, distdims, comm_size)
    else:
        try:
            grid_shape = tuple(grid_shape)
        except:
            raise InvalidGridShapeError("grid_shape not castable to a tuple")
    if len(grid_shape)!=ndistdim:
        raise InvalidGridShapeError("grid_shape has the wrong length")
    ngriddim = reduce(lambda x,y: x*y, grid_shape)
    if ngriddim != comm_size:
        raise InvalidGridShapeError("grid_shape is incompatible with the number of processors")
    return grid_shape


def optimize_grid_shape(shape, grid_shape, distdims, comm_size):
    ndistdim = len(distdims)
    if ndistdim==1:
        grid_shape = (comm_size,)
    else:
        factors = utils.mult_partitions(comm_size, ndistdim)
        if factors != []:
            reduced_shape = [shape[i] for i in distdims]
            factors = [utils.mirror_sort(f, reduced_shape) for f in factors]
            rs_ratio = _compute_grid_ratios(reduced_shape)
            f_ratios = [_compute_grid_ratios(f) for f in factors]
            distances = [rs_ratio-f_ratio for f_ratio in f_ratios]
            norms = np.array([np.linalg.norm(d,2) for d in distances])
            index = norms.argmin()
            grid_shape = tuple(factors[index])
        else:
            raise GridShapeError("Cannot distribute array over processors")
    return grid_shape


def _compute_grid_ratios(shape):
    n = len(shape)
    return np.array([float(shape[i])/shape[j] for i in range(n) for j in range(n) if i < j])


def init_comm(base_comm, grid_shape, ndistdim):
    return base_comm.Create_cart(grid_shape,ndistdim*(False,),False)


def init_local_shape_and_maps(shape, grid_shape, distdims, map_classes):
    maps = []
    local_shape = list(shape)
    for i, distdim in enumerate(distdims):
        minst = map_classes[i](shape[distdim], grid_shape[i])
        local_shape[distdim]= minst.local_shape
        maps.append(minst)
    return tuple(local_shape), tuple(maps)


def find_local_shape(shape, dist={0:'b'}, grid_shape=None, comm_size=None):
    if comm_size is None:
        raise ValueError("comm_size can't be None")
    ndim = len(shape)
    dist = init_dist(dist, ndim)
    distdims = init_distdims(dist, ndim)
    ndistdim = len(distdims)
    map_classes = init_map_classes(dist)   
    grid_shape = init_grid_shape(shape, grid_shape, distdims, comm_size)
    local_shape, maps = init_local_shape_and_maps(shape, 
        grid_shape, distdims, map_classes)
    return local_shape


def find_grid_shape(shape, dist={0:'b'}, grid_shape=None, comm_size=None):
    if comm_size is None:
        raise ValueError("comm_size can't be None")
    ndim = len(shape)
    dist = init_dist(dist, ndim)
    distdims = init_distdims(dist, ndim)
    ndistdim = len(distdims)
    map_classes = init_map_classes(dist)   
    grid_shape = init_grid_shape(shape, grid_shape, distdims, comm_size)
    return grid_shape

