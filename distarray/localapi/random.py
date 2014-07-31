# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

""" Pseudo-random number generation routines for local arrays.

This module provides a number of routines for generating random numbers,
from a variety of probability distributions.
"""

from hashlib import sha256
import numpy as np

from distarray.localapi.localarray import LocalArray


def label_state(comm):
    """ Label/personalize the random generator state for the local rank.

    This ensures that each separate engine, when using the same global seed,
    will generate a different sequence of pseudo-random numbers.
    """

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
    """ Return an array with random numbers from the beta probability distribution.

    Parameters
    ----------
    a: float
        Parameter that describes the beta probability distribution.
    b: float
        Parameter that describes the beta probability distribution.
    distribution: The desired distribution of the array.
        If None, then a normal NumPy array is returned.
        Otherwise, a LocalArray with this distribution is returned.

    Returns
    -------
    An array with random numbers.
    """
    if distribution is None:
        return np.random.beta(a, b)
    else:
        dtype = np.random.beta(a, b, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.beta(a, b, size=la.local_shape)
        return la


def normal(loc=0.0, scale=1.0, distribution=None):
    """ Return an array with random numbers from a normal (Gaussian) probability distribution.

    Parameters
    ----------
    loc: float
        The mean (or center) of the probability distribution.
    scale: float
        The standard deviation (or width) of the probability distribution.
    distribution: The desired distribution of the array.
        If None, then a normal NumPy array is returned.
        Otherwise, a LocalArray with this distribution is returned.

    Returns
    -------
    An array with random numbers.
    """
    if distribution is None:
        return np.random.normal(loc, scale)
    else:
        dtype = np.random.normal(loc, scale, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.normal(loc, scale, size=la.local_shape)
        return la


def rand(distribution=None):
    """ Return an array with random numbers distributed over the interval [0, 1).

    Parameters
    ----------
    distribution: The desired distribution of the array.
        If None, then a normal NumPy array is returned.
        Otherwise, a LocalArray with this distribution is returned.

    Returns
    -------
    An array with random numbers.
    """
    if distribution is None:
        return np.random.rand()
    else:
        dtype = np.random.rand(1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.rand(*la.local_shape)
        return la


def randint(low, high=None, distribution=None):
    """ Return random integers from low (inclusive) to high (exclusive).

    Return random integers from the “discrete uniform” distribution in the “half-open” interval [low, high).
    If high is None (the default), then results are from [0, low).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless high=None, in which case this parameter is the highest such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None).
    distribution: The desired distribution of the array.
        If None, then a normal NumPy array is returned.
        Otherwise, a LocalArray with this distribution is returned.

    Returns
    -------
    An array with random numbers.
    """
    if distribution is None:
        return np.random.randint(low, high)
    else:
        dtype = np.random.randint(low, high, size=1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.randint(low, high, size=la.local_shape)
        return la


def randn(distribution=None):
    """ Return a sample (or samples) from the “standard normal” distribution.

    Parameters
    ----------
    distribution: The desired distribution of the array.
        If None, then a normal NumPy array is returned.
        Otherwise, a LocalArray with this distribution is returned.

    Returns
    -------
    An array with random numbers.
    """
    if distribution is None:
        return np.random.randn()
    else:
        dtype = np.random.randn(1).dtype
        la = LocalArray(distribution, dtype=dtype)
        la.ndarray[:] = np.random.randn(*la.local_shape)
        return la
