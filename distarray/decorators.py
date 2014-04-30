# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Decorators for defining functions that use `DistArrays`.
"""

import functools

from distarray.client import DistArray
from distarray.context import DISTARRAY_BASE_NAME
from distarray.error import ContextError
from distarray.externals.six import string_types


class FunctionRegistrationBase(object):
    """
    Base class for local function registration.

    Subclasses:
        Localize
        Vectorize

    """

    def __init__(self, func, context):
        self.func = func
        self.func_key = self.func.__name__
        functools.update_wrapper(self, func)
        self.context = context

    def determine_context(self, args, kwargs):
        """ Determine a context from a functions arguments."""

        # inspect args for a context
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, DistArray):
                if arg.context != self.context:
                    msg = "DistArray %r not in same context as registered function %r."
                    raise ContextError(msg % (arg, self.func))

        return self.context

    def build_args(self, args, kwargs):
        """
        Returns a new args tuple and kwargs dictionary with all distarrays in
        the original args and kwargs arguments replaced by their .keys.

        """

        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, DistArray):
                args[idx] = arg.key

        # handle key word arguments
        for k, v in kwargs.items():
            if isinstance(v, DistArray):
                kwargs[k] = v.key

        return args, kwargs

    def process_return_value(self, result_from_target):
        """Figure out what to return on the Client.

        Parameters
        ----------
        key : string
            Key corresponding to wrapped function's return value.

        Returns
        -------
        Varied
            A DistArray (if locally all values are DistArray), a None (if
            locally all values are None), or else, pull the result back to the
            client and return it.  If all but one of the pulled values is None,
            return that non-None value only.
        """

        results = list(result_from_target.values())

        if all(isinstance(r, string_types) and r.startswith(DISTARRAY_BASE_NAME)
                for r in results):
            result = DistArray.from_localarrays(results[0], self.context)
        elif all(r is None for r in results):
            result = None
        else:
            non_nones = [r for r in results if r is not None]
            if len(non_nones) == 1:
                result = non_nones[0]
            else:
                result = results
        return result


def _rpc_localize(func, args, kwargs, result_key, prefix):

    ns = __import__('__main__')

    from distarray.local.localarray import LocalArray
    from distarray.externals.six import string_types

    args = list(args)
    for idx, a in enumerate(args):
        if isinstance(a, string_types):
            if a.startswith(prefix):
                args[idx] = getattr(ns, a)

    for k, v in kwargs.items():
        if isinstance(v, string_types):
            if v.startswith(prefix):
                kwargs[k] = getattr(ns, v)

    res = func(*args, **kwargs)
    if isinstance(res, LocalArray):
        setattr(ns, result_key, res)
        return result_key
    return res


class Localize(FunctionRegistrationBase):
    """Runs a function locally on the engines."""

    def __call__(self, *args, **kwargs):
        context = self.determine_context(args, kwargs)
        args, kwargs = self.build_args(args, kwargs)
        result_key = context._generate_key()
        results = context.view.apply_async(_rpc_localize, self.func,
                                           args, kwargs, result_key,
                                           DISTARRAY_BASE_NAME).get_dict()
        return self.process_return_value(results)


def _rpc_vectorize(func, args, kwargs, out, prefix):

    ns = __import__('__main__')
    import numpy as np
    from distarray.externals.six import string_types

    args = list(args)
    for idx, a in enumerate(args):
        if isinstance(a, string_types):
            if a.startswith(prefix):
                args[idx] = getattr(ns, a).local_array

    for k, v in kwargs.items():
        if isinstance(v, string_types):
            if v.startswith(prefix):
                kwargs[k] = getattr(ns, v).local_array

    out = getattr(ns, out)

    func = np.vectorize(func)
    out.local_array = func(*args)


class Vectorize(FunctionRegistrationBase):
    """
    Like `Localize`, but vectorizes the function with numpy.vectorize and runs
    it on the engines.
    """

    def __call__(self, *args, **kwargs):
        context = self.determine_context(args, kwargs)
        # TODO: FIXME: This uses an extra round-trip (or two (or three)) to
        # create the `out` array.  Better would be to create a new LocalArray
        # inside _rpc_vectorize and return its metadata to create a DistArray
        # using `.from_localarrays()`.
        for arg in args:
            if isinstance(arg, DistArray):
                out = context.empty(arg.shape, dtype=arg.dtype,
                                    dist=arg.dist, grid_shape=arg.grid_shape)
        args, kwargs = self.build_args(args, kwargs)
        context.view.apply_sync(_rpc_vectorize, self.func,
                                 args, kwargs, out.key, DISTARRAY_BASE_NAME)
        return out
