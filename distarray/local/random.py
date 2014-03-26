# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import hashlib
import numpy as np
from six import indexbytes, int2byte

from distarray.local.localarray import LocalArray


def label_state(comm):
    """ Label/personalize the random generator state for the local rank."""

    def get_mask(rank):
        """ Get a uint32 mask to use to xor the random generator state.

        We do not simply return the rank, as this small change to the
        state of the Mersenne Twister only causes small changes in
        the generated sequence. (The generators will eventually
        diverge, but this takes a while.) So we scramble the mask up
        a lot more, still deterministically, using a cryptographic hash.
        See: http://en.wikipedia.org/wiki/Mersenne_twister#Disadvantages
        """
        m = hashlib.sha256()
        m.update(int2byte(rank))
        # Construct a uint32 from the start of the digest.
        digest = m.digest()
        value = 0
        for i in range(4):
            value += indexbytes(digest, i) << (8 * i)
        mask = np.array([value], dtype=np.uint32)
        return mask

    rank = comm.Get_rank()
    mask = get_mask(rank)
    # For the Mersenne Twister used by numpy, the state is a 5-tuple,
    # with the important part being an array of 624 uint32 values.
    # We xor the mask into that array, and leave the rest of the tuple alone.
    s0, orig_array, s2, s3, s4 = np.random.get_state()
    mod_array = np.bitwise_xor(orig_array, mask)
    np.random.set_state((s0, mod_array, s2, s3, s4))


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
