# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Functions to validate Distributed Array Protocol data structures for protocol
version 0.10.0.

Functions
---------
validate_dim_dict(idx, dim_dict)
validate_dim_data(dim_data)
validate(distbuffer)

"""

from distutils.version import StrictVersion


VERSIONS = ((0, 10, 0),)


def _verify_exact_keys(dd, keys):
    dkeys = set(dd)
    keys = set(keys)
    missing = keys - dkeys
    extra = dkeys - keys
    return (extra, missing)


def _validate_common_dist_keys(idx, dim_dict):
    # Verify presence of common keys for distributed dimensions.
    extra, missing = _verify_exact_keys(dim_dict, 'proc_grid_rank proc_grid_size'.split())
    if missing:
        msg = 'keys %r missing for dimension %d.'
        return (False, msg % (tuple(missing), idx))

    # Verify process grid keys.
    proc_grid_size = dim_dict['proc_grid_size']
    proc_grid_rank = dim_dict['proc_grid_rank']

    if not isinstance(proc_grid_size, int):
        msg = 'proc_grid_size for dimension %d has non-integer type %r.'
        return (False, msg % (idx, type(proc_grid_size)))

    if not isinstance(proc_grid_rank, int):
        msg = 'proc_grid_rank for dimension %d has non-integer type %r.'
        return (False, msg % (idx, type(proc_grid_rank)))

    if proc_grid_size < 1:
        msg = 'proc_grid_size for dimension %d has value %d and is less than one.'
        return (False, msg % (idx, proc_grid_size))

    if proc_grid_rank >= proc_grid_size:
        msg = 'proc_grid_rank (value %d) >= proc_grid_size (value %d) for dimension %d.'
        return (False, msg % (proc_grid_rank, proc_grid_size, idx))

    return (True, '')

def _validate_padding(padding, idx):
    if not isinstance(padding, tuple):
        msg = 'padding (%r) for dimension %d is not a tuple.'
        return (False, msg % (padding, idx))
    if not len(padding) == 2:
        msg = 'padding (%r) for dimension %d is not of length 2.'
        return (False, msg % (padding, idx))
    for i, element in enumerate(padding):
        if not isinstance(element, int):
            msg = 'padding element %r (%r) for dimension %d has non-integer type %r.'
            return (False, msg % (i, padding[i], idx, type(element)))

    return (True, '')

def _validate_block(idx, dim_dict):
    """Verify block distribution."""
    if dim_dict['dist_type'] != 'b':
        msg = '_validate_block expecting a dist_type of "b". (given %r)'
        raise ValueError(msg % dim_dict['dist_type'])
    is_valid, msg = _validate_common_dist_keys(idx, dim_dict)
    if not is_valid:
        return (is_valid, msg)
    for k in 'start stop'.split():
        if k not in dim_dict:
            msg = 'key %r not present in dictionary for distributed dimension %d.'
            return (False, msg % (k, idx))
        if not isinstance(dim_dict[k], int):
            msg = '%r for dimension %d has non-integer type %r.'
            return (False, msg % (k, idx, type(dim_dict[k])))
    start = dim_dict['start']
    stop = dim_dict['stop']
    if start < 0:
        msg = 'start for dimension %d has negative value %d.'
        return (False, msg % (idx, start))
    if stop < start:
        msg = 'stop (%d) for dimension %d is less than start (%d).'
        return (False, msg % (stop, idx, start))
    size = dim_dict['size']
    if stop > size:
        msg = 'stop (%d) for dimension %d is greater than size (%d).'
        return (False, msg % (stop, idx, size))


    # Validate padding.
    padding = dim_dict.get('padding', (0,0))
    is_valid, msg = _validate_padding(padding, idx)
    if not is_valid:
        return (is_valid, msg)

    # Validate start.
    is_valid, msg = _validate_start(idx, dim_dict)
    if not is_valid:
        return (is_valid, msg)

    return (True, '')

def _validate_cyclic(idx, dim_dict):
    """Verify cyclic (and block-cyclic) distribution."""
    if dim_dict['dist_type'] != 'c':
        msg = '_validate_cyclic expecting a dist_type of "c". (given %r)'
        raise ValueError(msg % dim_dict['dist_type'])
    is_valid, msg = _validate_common_dist_keys(idx, dim_dict)
    if not is_valid:
        return (is_valid, msg)
    if 'start' not in dim_dict:
        msg = 'start not present in dictionary for distributed dimension %d.'
        return (False, msg % idx)
    start = dim_dict['start']
    if not isinstance(start, int):
        msg = 'start for dimension %d has non-integer type %r.'
        return (False, msg % (idx, type(start)))
    block_size = dim_dict.get('block_size', 1)
    if not isinstance(block_size, int):
        msg = 'block_size for dimension %d has non-integer type %r.'
        return (False, msg % (idx, type(block_size)))
    if block_size < 1:
        msg = 'block_size (%d) for dimension %d is less than one.'
        return (False, msg % (block_size, idx))
    proc_grid_rank = dim_dict['proc_grid_rank']
    if proc_grid_rank * block_size != start:
        msg = 'proc_grid_rank * block_size != start (%d * %d != %d).'
        return (False, msg % (proc_grid_rank, block_size, start))
    is_valid, msg = _validate_start(idx, dim_dict)
    if not is_valid:
        return (is_valid, msg)

    return (True, '')

def _validate_start(idx, dim_dict):
    # Verify that start == 0 when proc_grid_rank == 0.
    dist_type = dim_dict['dist_type']
    proc_grid_rank = dim_dict['proc_grid_rank']
    start = dim_dict['start']
    if dist_type in ('b', 'c') and proc_grid_rank == 0 and start != 0:
        msg = 'start should be zero, but has value %d for dimension %d with proc_grid_rank == %d.'
        return (False, msg % (start, idx, proc_grid_rank))

    return (True, '')

def _validate_unstructured(idx, dim_dict):
    """Verify unstructured distribution."""
    if dim_dict['dist_type'] != 'u':
        msg = '_validate_unstructured expecting a dist_type of "u". (given %r)'
        raise ValueError(msg % dim_dict['dist_type'])
    is_valid, msg = _validate_common_dist_keys(idx, dim_dict)
    if not is_valid:
        return (is_valid, msg)
    if 'indices' not in dim_dict:
        msg = 'indices not present in dictionary for distributed dimension %d.'
        return (False, msg % idx)
    try:
        memoryview(dim_dict['indices'])
    except TypeError:
        msg = 'indices object of type %r for distributed dimension %d does not support the buffer interface.'
        return (False, msg % (type(dim_dict['indices']), idx))
    one_to_one = dim_dict.get('one_to_one', False)
    if not isinstance(one_to_one, bool):
        msg = 'one_to_one for dimension %d has non-boolean type %r.'
        return (False, msg % (idx, type(one_to_one)))

    return (True, '')

def _validate_common(idx, dim_dict):
    # Verify presence of 'dist_type' and 'size' keys.
    extra, missing = _verify_exact_keys(dim_dict, 'dist_type size'.split())
    if missing:
        msg = 'keys %r missing for dimension %d.'
        return (False, msg % (tuple(missing), idx))

    # Verify size.
    size = dim_dict['size']

    if not isinstance(size, int):
        msg = 'size for dimension %d has non-integer type %r.'
        return (False, msg % (idx, type(size)))

    if size < 0:
        msg = 'size for dimension %d has negative value %d.'
        return (False, msg % (idx, size))

    # Verify dist_type.
    dist_type = dim_dict['dist_type']
    if dist_type not in set('bcu'):
        msg = 'dimension dictionary for dimension %d has an invalid dist_type of %r'
        return (False, msg % (idx, dist_type))

    # Verify optional periodic key.
    periodic = dim_dict.get('periodic', False)
    if not isinstance(periodic, bool):
        msg = 'Optional periodic key is present for dimension %d but has non-boolean value %r.'
        return (False, msg % (idx, periodic))

    return (True, '')


def validate_dim_dict(idx, dim_dict):
    """
    Validate that `dim_dict` conforms to the Distributed Array Protocol.

    Returns a 2-tuple of a boolean and string; boolean indicates validity, the
    string indicates the reason for invalidity, empty otherwise.

    """
    # Check for the empty dim_dict alias
    if not dim_dict: # the dim_dict is empty
        return (True, '')

    dist_type = dim_dict['dist_type']

    if dist_type not in set('bcu'):
        msg = "dimension dict at index %r should have dist_type of 'b', 'c', or 'u' (given %r)"
        return (False, msg % (idx, dist_type))

    # Select the appropriate distribution type and validate.
    return {'b': _validate_block,
            'c': _validate_cyclic,
            'u': _validate_unstructured,
            }[dist_type](idx, dim_dict)


def validate_dim_data(dim_data):
    """
    Validate that `dim_data` conforms to the Distributed Array Protocol.

    Returns a 2-tuple of a boolean and string; boolean indicates validity, the
    string indicates the reason for invalidity, empty otherwise.

    """
    # First, check that it's a tuple...
    if not isinstance(dim_data, tuple):
        msg = 'dim_data (type %r) is not an instance of tuple.'
        return (False, msg % type(dim_data))

    # ...of dictionaries.
    for idx, dim_dict in enumerate(dim_data):
        if not isinstance(dim_dict, dict):
            msg = 'object at index %d with type %r is not an instance of dict.'
            return (False, msg % (idx, type(dim_dict)))

    # Verify each dim_data dictionary.
    for idx, dim_dict in enumerate(dim_data):
        is_valid, msg = validate_dim_dict(idx, dim_dict)
        if not is_valid:
            return (is_valid, msg)

    return (True, '')


def validate(distbuffer):
    """
    Validate that `distbuffer` conforms to the Distributed Array Protocol.

    Returns a 2-tuple of a boolean and string; boolean indicates validity, the
    string indicates the reason for invalidity, empty otherwise.

    """
    # Verify distbuffer is a dictionary.
    if not isinstance(distbuffer, dict):
        msg = 'non-dictionary object of type %r.'
        return (False, msg % type(distbuffer))

    # Verify distbuffer keys.
    extra, missing = _verify_exact_keys(distbuffer,
            '__version__ buffer dim_data'.split())
    if extra:
        msg = 'extra keys %r present.'
        return (False, msg % (tuple(extra),))
    if missing:
        msg = 'keys %r missing.'
        return (False, msg % (tuple(missing),))

    # Verify the version string.
    version = distbuffer['__version__']
    try:
        strict_version = StrictVersion(version)
    except ValueError:
        msg = '__version__ "%s" not valid.'
        return (False, msg % version)

    # Verify the version number.
    if strict_version.version not in VERSIONS:
        msg = '__version__ "%s" not supported by this checker.'
        return (False, msg % version)

    # Verify the buffer.
    buffer = distbuffer['buffer']
    try:
        buffer = memoryview(buffer)
    except TypeError:
        msg = '%r does not have the buffer interface.'
        return (False, msg % buffer)

    # Verify the dim_data tuple.
    dim_data = distbuffer['dim_data']
    is_valid, msg = validate_dim_data(dim_data)
    if not is_valid:
        return (is_valid, msg)

    # Verify the number of dim_data dictionaries matches the number of
    # dimensions in the buffer.
    # (This can't be done in validate_dim_data because it doesn't have access
    # to the buffer.)
    if len(dim_data) != buffer.ndim:
        msg = 'len(dim_data) == %d, which is not equal to buffer.ndim == %d.'
        return (False, msg % (len(dim_data), buffer.ndim))

    return (True, '')

