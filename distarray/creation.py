# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------
"""
DistArray creation functions.
"""

from functools import wraps

import numpy

from distarray.client import DistArray


def default_context(func):
    """Importing `from distarray.world import WORLD` at the module
    level results in circular imports. So we do this, to import it
    lazily.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('context') is None:
            from distarray.world import WORLD
            kwargs['context'] = WORLD
        return func(*args, **kwargs)
    return wrapper


def _create_local(context, local_call, shape, dtype, dist, grid_shape):
    """Creates a local array, according to the method named in `local_call`."""
    keys = context._key_and_push(shape, dtype, dist, grid_shape)
    shape_name, dtype_name, dist_name, grid_shape_name = keys
    da_key = context._generate_key()
    comm = context._comm_key
    cmd = ('{da_key} = {local_call}({shape_name}, {dtype_name}, {dist_name}, '
           '{grid_shape_name}, {comm})')
    context._execute(cmd.format(**locals()))
    return DistArray.from_localarrays(da_key, context)


@default_context
def from_dim_data(dim_data_per_rank, context=None, dtype=float):
    """Make a DistArray from dim_data structures.

    Parameters
    ----------
    dim_data_per_rank : sequence of tuples of dict
        A "dim_data" data structure for every rank.  Described here:
        https://github.com/enthought/distributed-array-protocol
    dtype : numpy dtype, optional
        dtype for underlying arrays

    Returns
    -------
    result : DistArray
        An empty DistArray of the specified size, dimensionality, and
        distribution.

    """
    if len(context.targets) != len(dim_data_per_rank):
        errmsg = "`dim_data_per_rank` must contain a dim_data for every rank."
        raise TypeError(errmsg)

    da_key = context._generate_key()
    subs = ((da_key,) + context._key_and_push(dim_data_per_rank) +
            (context._comm_key,) + context._key_and_push(dtype) +
            (context._comm_key,))

    cmd = ('%s = distarray.local.LocalArray.'
           'from_dim_data(%s[%s.Get_rank()], dtype=%s, comm=%s)')
    context._execute(cmd % subs)

    return DistArray.from_localarrays(da_key, context)


@default_context
def zeros(shape, context=None, dtype=float, dist={0: 'b'}, grid_shape=None):
    return _create_local(context, local_call='distarray.local.zeros',
                         shape=shape, dtype=dtype, dist=dist,
                         grid_shape=grid_shape)


@default_context
def ones(shape, context=None, dtype=float, dist={0: 'b'}, grid_shape=None):
    return _create_local(context, local_call='distarray.local.ones',
                         shape=shape, dtype=dtype, dist=dist,
                         grid_shape=grid_shape)


@default_context
def empty(shape, context=None, dtype=float, dist={0: 'b'}, grid_shape=None):
    return _create_local(context, local_call='distarray.local.empty',
                         shape=shape, dtype=dtype, dist=dist,
                         grid_shape=grid_shape)


@default_context
def fromndarray(arr, context=None, dist={0: 'b'}, grid_shape=None):
    """Convert an ndarray to a distarray."""
    out = empty(arr.shape, dtype=arr.dtype, dist=dist, grid_shape=grid_shape)
    for index, value in numpy.ndenumerate(arr):
        out[index] = value
    return out

fromarray = fromndarray


@default_context
def fromfunction(function, shape, context=None, **kwargs):
    func_key = context._generate_key()
    context.view.push_function({func_key: function}, targets=context.targets,
                               block=True)
    keys = context._key_and_push(shape, kwargs)
    new_key = context._generate_key()
    subs = (new_key, func_key) + keys
    context._execute('%s = distarray.local.fromfunction(%s,%s,**%s)' % subs)
    return DistArray.from_localarrays(new_key, context)
