# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from hashlib import sha256
import numpy as np

from distarray.local.localarray import LocalArray


def label_state(comm):
    """ Label/personalize the random generator state for the local rank."""

    def get_mask(rank):
        """ Get a uint32 mask array to use to xor the random generator state.

        We do not simply return the rank, as this small change to the
        state of the Mersenne Twister only causes small changes in
        the generated sequence. (The generators will eventually
        diverge, but this takes a while.) So we scramble the mask up
        a lot more, still deterministically, using a cryptographic hash.
        See: http://en.wikipedia.org/wiki/Mersenne_twister#Disadvantages
        """
        # Since we will be converting to/from bytes, endianness is important.
        uint32be = np.dtype('>u4')
        x = np.empty([624 // 8, 2], dtype=uint32be)
        # The hash of the rank catted with an increasing index
        # (stuffed into big-endian uint32s) are hashed with SHA-256 to make
        # the XOR mask for 8 consecutive uint32 words for a 624-word
        # Mersenne Twister state.
        x[:, 0] = rank
        x[:, 1] = np.arange(624 // 8)
        mask_buffer = b''.join(sha256(row).digest() for row in x)
        # And convert back to native-endian.
        mask = np.frombuffer(mask_buffer, dtype=uint32be).astype(np.uint32)
        return mask

    rank = comm.Get_rank()
    mask = get_mask(rank)
    # For the Mersenne Twister used by numpy, the state is a 5-tuple,
    # with the important part being an array of 624 uint32 values.
    # We xor the mask into that array, and leave the rest of the tuple alone.
    s0, orig_array, s2, s3, s4 = np.random.get_state()
    mod_array = np.bitwise_xor(orig_array, mask)
    np.random.set_state((s0, mod_array, s2, s3, s4))


def beta(a, b, distribution=None):
    if distribution is None:
        return np.random.beta(a, b)
    else:
        dtype = np.random.beta(a, b, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.beta(a, b, size=la.local_shape)
        return la


def normal(loc=0.0, scale=1.0, distribution=None):
    if distribution is None:
        return np.random.normal(loc, scale)
    else:
        dtype = np.random.normal(loc, scale, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.normal(loc, scale, size=la.local_shape)
        return la


def rand(distribution=None):
    if distribution is None:
        return np.random.rand()
    else:
        dtype = np.random.rand(1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.rand(*la.local_shape)
        return la


def randint(low, high=None, distribution=None):
    if distribution is None:
        return np.random.randint(low, high)
    else:
        dtype = np.random.randint(low, high, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.randint(low, high, size=la.local_shape)
        return la


def randn(distribution=None):
    if distribution is None:
        return np.random.randn()
    else:
        dtype = np.random.randn(1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.randn(*la.local_shape)
        return la
