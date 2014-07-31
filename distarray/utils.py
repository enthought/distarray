# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Utilities.
"""

from functools import reduce
from importlib import import_module
from math import sqrt
import random
import uuid

from distarray.externals.six import next

DISTARRAY_BASE_NAME = '__distarray__'
DISTARRAY_RANDOM = random.Random()


def distarray_random_setstate(state):
    """ Set the state of the global random number generator. """
    global DISTARRAY_RANDOM
    DISTARRAY_RANDOM.setstate(state)


def distarray_random_getstate():
    """ Get the state of the global random number generator. """
    global DISTARRAY_RANDOM
    return DISTARRAY_RANDOM.getstate()


def uid():
    """ Get a unique name for a distarray object. """
    suffix = uuid.UUID(int=DISTARRAY_RANDOM.getrandbits(8 * 16)).hex[:16]
    return DISTARRAY_BASE_NAME + suffix


def multi_for(iterables):
    if not iterables:
        yield ()
    else:
        for item in iterables[0]:
            for rest_tuple in multi_for(iterables[1:]):
                yield (item,) + rest_tuple


def divisors_minmax(n, dmin, dmax):
    """Find the divisors of n in the interval (dmin,dmax]."""
    i = dmin + 1
    while i <= dmax:
        if n % i == 0:
            yield i
        i += 1


def list_or_tuple(seq):
    """ Is the object either a list or a tuple? """
    return isinstance(seq, (list, tuple))


def flatten(seq, to_expand=list_or_tuple):
    """Flatten a nested sequence."""
    for item in seq:
        if to_expand(item):
            for subitem in flatten(item, to_expand):
                yield subitem
        else:
            yield item


def mult_partitions(n, s):
    """Compute the multiplicative partitions of n of size s

    >>> mult_partitions(52,3)
    [(2, 2, 13)]
    >>> mult_partitions(52,2)
    [(2, 26), (4, 13)]
    """
    return [tuple(flatten(p)) for p in mult_partitions_recurs(n, s)]


def mult_partitions_recurs(n, s, pd=0):
    if s == 1:
        return [n]
    divs = divisors_minmax(n, pd, int(sqrt(n)))
    fs = []
    for d in divs:
        fs.extend([(d, f) for f in mult_partitions_recurs(n / d, s - 1, pd)])
        pd = d
    return fs


def mirror_sort(seq, ref_seq):
    """Sort `seq` into the order that `ref_seq` is in.

    >>> mirror_sort(range(5),[1,5,2,4,3])
    [0, 4, 1, 3, 2]
    """
    if not len(seq) == len(ref_seq):
        raise ValueError("Sequences must have the same length")

    shift = list(zip(range(len(ref_seq)), ref_seq))
    shift.sort(key=lambda x: x[1])
    shift = [s[0] for s in shift]
    newseq = len(ref_seq) * [0]
    for s_index in range(len(shift)):
        newseq[shift[s_index]] = seq[s_index]
    return newseq


def _raise_nie():
    msg = "This has not yet been implemented for distributed arrays"
    raise NotImplementedError(msg)


def slice_intersection(s1, s2):
    """Compute a slice that represents the intersection of two slices.

    Currently only implemented for steps of size 1.

    Parameters
    ----------
    s1, s2 : slice objects

    Returns
    -------
    slice object
    """
    valid_steps = {None, 1}
    if (s1.step in valid_steps) and (s2.step in valid_steps):
        step = 1
        stop = min(s1.stop, s2.stop)
        start = max(s1.start, s2.start)
        return slice(start, stop, step)
    else:
        msg = "Slice intersection only implemented for step=1."
        raise NotImplementedError(msg)


def has_exactly_one(iterable):
    """Does `iterable` have exactly one non-None element?"""
    test = (x is not None for x in iterable)
    if sum(test) == 1:
        return True
    else:
        return False


def all_equal(iterable):
    """Return True if all elements in `iterable` are equal.

    Also returns True if iterable is empty.
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True  # vacuously True

    return all(element == first for element in iterator)


class count_round_trips(object):
    """
    Context manager for counting the number of roundtrips between a IPython
    client and controller.

    Usage:
        >>> with count_round_trips(client) as r:
        ...     send_42_messages()

        >>> r.count
        42
    """

    def __init__(self, client):
        self.client = client
        self.orig_count = len(self.client.history)
        self.count = 0

    def update_count(self):
        self.count = len(self.client.history) - self.orig_count

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.update_count()


def remove_elements(to_remove, seq):
    """ Return a list, with the elements with specified indices removed.

    Parameters
    ----------
    to_remove: iterable
        Indices of elements in list to remove
    seq: iterable
        Elements in the list.

    Returns
    -------
    List with the specified indices removed.
    """
    return [x for (idx, x) in enumerate(seq) if idx not in to_remove]


def get_from_dotted_name(dotted_name):
    main = import_module('__main__')
    thing = reduce(getattr, [main] + dotted_name.split('.'))
    return thing


def set_from_dotted_name(name, val):
    main = import_module('__main__')
    peices = name.split('.')
    place = reduce(getattr, [main] + peices[:-1])
    setattr(place, peices[-1], val)
