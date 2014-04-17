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
from distarray.context import Context, DISTARRAY_BASE_NAME
from distarray.error import ContextError
from distarray.utils import has_exactly_one

def _rpc(func, args, kwargs, result_key):

    args = list(args)
    for idx, a in enumerate(args):
        if isinstance(a, basestring):
            if a.startswith(DISTARRAY_BASE_NAME):
                args[idx] = eval(a)

    for k, v in kwargs.items():
        if isinstance(v, basestring):
            if v.startswith(DISTARRAY_BASE_NAME):
                kwargs[k] = eval(v)

    res = func(*args, **kwargs)
    if isinstance(res, LocalArray):
        globals()[result_key] = res
        res = result_key
    return res


class DecoratorBase(object):
    """
    Base class for decorators, handles name wrapping and allows the
    decorator to take an optional kwarg.
    """

    def __init__(self, fn, context):
        self.fn = fn
        self.fn_key = self.fn.__name__
        functools.update_wrapper(self, fn)
        self.context = context
        self.push_fn()

    def push_fn(self):
        """Push function to the engines."""
        self.context._push({self.fn_key: self.fn})

    def check_contexts(self, args, kwargs):
        """ Determine a context from a functions arguments."""

        # inspect args for a context
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, DistArray):
                if arg.context != self.context:
                    msg = "DistArray %r not in same context as registered function %r."
                    raise ContextError(msg % (arg, self.fn))

        return self.context

    def build_args(self, args, kwargs):
        """
        Push a tuple of args and dict of kwargs to the engines. Return a
        tuple with keys corresponding to args values on the engines. And a
        dictionary with the same keys and values which are the keys to the
        input dictionary's values.

        This allows us to use the following interface to execute code on
        the engines:

        >>> def foo(*args, **kwargs):
        >>>     args, kwargs = _key_and_push_args(args, kwargs)
        >>>     exec_str = "remote_foo(*%s, **%s)"
        >>>     exec_str %= (args, kwargs)
        >>>     context.execute(exec_str)
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
        # type_key = self.context._generate_key()
        # type_statement = "{} = str(type({}))".format(type_key, result_key)
        # context._execute(type_statement)
        # result_type_str = context._pull(type_key)

        results = list(result_from_target.values())

        if all(isinstance(r, basestring) and r.startswith(DISTARRAY_BASE_NAME)
                for r in results):
            result = DistArray.from_localarrays(results[0], self.context)
        elif all(r is None for r in results):
            result = None
        else:
            non_nones = [r for r in results if r is not None]
            if len(non_nones) == 1:
                result = non_nones[0]
        return result


class local(DecoratorBase):
    """Decorator to run a function locally on the engines."""

    def __call__(self, *args, **kwargs):
        # get context from args
        context = self.determine_context(args, kwargs)
        args, kwargs = self.build_args(args, kwargs)
        result_key = context._generate_key()
        results = context.view.apply_async(_rpc, self.func, args, kwargs, result_key).get_dict()
        return self.process_return_value(results)


class _vectorize(DecoratorBase):
    """
    Analogous to numpy.vectorize. Input DistArray's must all be the
    same shape, and this will be the shape of the output distarray.
    """

    def get_local_array(self, da, arg_keys):
        return arg_keys + [da.key + '.local_array']

    def __call__(self, *args, **kwargs):
        # get context from args
        context = self.determine_context(args, kwargs)
        # push function
        self.push_fn(context, self.fn_key, self.fn)
        # vectorize the function
        exec_str = "%s = numpy.vectorize(%s)" % (self.fn_key, self.fn_key)
        context._execute(exec_str)

        # Find the first distarray, they should all be the same up to the data.
        for arg in args:
            if isinstance(arg, DistArray):
                # Create the output distarray.
                out = context.empty(arg.shape, dtype=arg.dtype,
                                         dist=arg.dist,
                                         grid_shape=arg.grid_shape)
                # parse args
                args_str, kwargs_str = self.key_and_push_args(
                    args, kwargs, context=context,
                    da_handler=self.get_local_array)

                # Call the function
                exec_str = ("if %s.local_array.size != 0: %s.local_array = "
                            "%s(*%s, **%s)")
                exec_str %= (out.key, out.key, self.fn_key, args_str,
                             kwargs_str)
                context._execute(exec_str)
                return out
