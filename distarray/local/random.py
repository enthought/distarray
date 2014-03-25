# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import numpy as np
from distarray.local.localarray import LocalArray


def label_state(comm):
    """ Modify/label the random generator state to include the local rank."""
    rank = comm.Get_rank()
    #print('rank:', rank)
    # We will mix in the rank value into the random generator state.
    # First, we xor each element of the state vector with the rank.
    # This is not sufficient to cause the sequences to be completely
    # different, so we also roll the array by the rank.
    # It also helps to 'burn' a number of random values,
    # to let the sequences diverge more completely.
    # See: http://en.wikipedia.org/wiki/Mersenne_twister#Disadvantages
    # The state is a 5-tuple, with the second part being an array.
    # We will mix up that array and leave the rest alone.
    s0, orig_array, s2, s3, s4 = np.random.get_state()
    mod_array = np.bitwise_xor(orig_array, rank)
    mod_array = np.roll(mod_array, rank)
    np.random.set_state((s0, mod_array, s2, s3, s4))
    # 'Burn' some numbers off the sequence to make them diverge more.
    _ = np.random.bytes(1024)


def beta(a, b, size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.beta(a, b)
    else:
        dtype = np.random.beta(a, b, size=1).dtype
        la = LocalArray(size, dtype=dtype, dist=dist, grid_shape=grid_shape,
                        comm=comm)
        la.local_array[:] = np.random.beta(a, b, size=la.local_shape)
        return la


def normal(loc=0.0, scale=1.0, size=None, dist=None, grid_shape=None,
           comm=None):
    if size is None:
        return np.random.normal(loc, scale)
    else:
        dtype = np.random.normal(loc, scale, size=1).dtype
        la = LocalArray(size, dtype=dtype, dist=dist, grid_shape=grid_shape,
                        comm=comm)
        la.local_array[:] = np.random.normal(loc, scale, size=la.local_shape)
        return la


def rand(size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.rand()
    else:
        dtype = np.random.rand(1).dtype
        la = LocalArray(size, dtype=dtype, dist=dist, grid_shape=grid_shape,
                        comm=comm)
        la.local_array[:] = np.random.rand(*la.local_shape)
        return la


def randint(low, high=None, size=None, dist=None, grid_shape=None,
            comm=None):
    if size is None:
        return np.random.randint(low, high)
    else:
        dtype = np.random.randint(low, high, size=1).dtype
        la = LocalArray(size, dtype=dtype, dist=dist, grid_shape=grid_shape,
                        comm=comm)
        la.local_array[:] = np.random.randint(low, high, size=la.local_shape)
        return la


def randn(size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.randn()
    else:
        dtype = np.random.randn(1).dtype
        la = LocalArray(size, dtype=dtype, dist=dist, grid_shape=grid_shape,
                        comm=comm)
        la.local_array[:] = np.random.randn(*la.local_shape)
        return la
