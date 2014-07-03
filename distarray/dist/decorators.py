# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Decorators for defining functions that use `DistArrays`.
"""

from __future__ import absolute_import

import functools

from distarray.dist.distarray import DistArray
from distarray.dist.maps import Distribution
from distarray.error import DistributionError
from distarray.utils import has_exactly_one


class DecoratorBase(object):
    """
    Base class for decorators, handles name wrapping and allows the
    decorator to take an optional kwarg.
    """

    def __init__(self, fn):
        self.fn = fn
        self.fn_key = self.fn.__name__
        functools.update_wrapper(self, fn)
        self.context = None

    def push_fn(self, context, fn_key, fn):
        """Push function to the engines."""
        context._push({fn_key: fn}, targets=context.targets)

    def determine_distribution(self, args, kwargs):
        """ Determine a distribution from a functions arguments."""

        dists = []
        # inspect args for a context
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, DistArray):
                dists.append(arg.distribution)
            elif isinstance(arg, Distribution):
                dists.append(arg)

        # check the args had a context
        if dists == []:
            raise TypeError('Function must take DistArray or Distribution'
                            ' objects.')

        # check that all contexts are equal
        if not dists.count(dists[0]) == len(dists):
            msg = ("Arguments must use the same Distribution (given arguments "
                   "of type %r)")
            msg %= (tuple(set(dists)),)
            raise DistributionError(msg)

        return dists[0]

    def key_and_push_args(self, args, kwargs, context=None, da_handler=None):
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

        if context is None:
            distribution = self.determine_distribution(args, kwargs)
            context = distribution.context

        # handle positional arguments
        arg_keys = []
        push_keys = {}
        for arg in args:
            if isinstance(arg, DistArray):
                if da_handler is None:
                    arg_keys.append(arg.key)
                # da_handler handles distarrays.
                else:
                    arg_keys = da_handler(arg, arg_keys)
            else:
                new_key = context._generate_key()
                arg_keys.append(new_key)
                push_keys[new_key] = arg

        # handle key word arguments
        for kw in kwargs:
            if isinstance(kwargs[kw], DistArray):
                kwargs[kw] = kwargs[kw].key
            else:
                new_key = context._generate_key()
                push_keys[new_key] = kwargs[kw]
                kwargs[kw] = new_key

        # push the keys to the engines
        context._push(push_keys, targets=context.targets)

        # build arg string
        arg_str = '(' + ', '.join(arg_keys) + ',)'

        # build kwarg string
        kwarg_iter = ["'%s': %s" % (k, v) for (k, v) in kwargs.items()]
        kwarg_str = '{' + ', '.join(kwarg_iter) + '}'

        return arg_str, kwarg_str

    def process_return_value(self, context, targets, result_key):
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
        def get_type_str(key):
            return str(type(key))
        result_type_str = context.apply(get_type_str, args=(result_key,),
                                        targets=targets)

        def is_NoneType(typestring):
            return (typestring == "<type 'NoneType'>" or
                    typestring == "<class 'NoneType'>")

        def is_LocalArray(typestring):
            return (typestring == "<class 'distarray.local.localarray."
                                  "LocalArray'>")

        if all(is_LocalArray(r) for r in result_type_str):
            result = DistArray.from_localarrays(result_key, context=context, targets=targets)
        elif all(is_NoneType(r) for r in result_type_str):
            result = None
        else:
            result = context._pull(result_key, targets=targets)
            if has_exactly_one(result):
                result = next(x for x in result if x is not None)

        return result


class local(DecoratorBase):
    """Decorator to run a function locally on the engines."""

    def __call__(self, *args, **kwargs):
        # get context from args
        distribution = self.determine_distribution(args, kwargs)
        context = distribution.context
        # push function
        self.push_fn(context, self.fn_key, self.fn)

        args, kwargs = self.key_and_push_args(args, kwargs,
                                              context=context)
        result_key = context._generate_key()

        exec_str = "%s = %s(*%s, **%s)"
        exec_str %= (result_key, self.fn_key, args, kwargs)
        context._execute(exec_str, targets=distribution.targets)

        return self.process_return_value(context, distribution.targets, result_key)


class vectorize(DecoratorBase):
    """
    Analogous to numpy.vectorize. Input DistArray's must all be the
    same shape, and this will be the shape of the output distarray.
    """

    def get_ndarray(self, da, arg_keys):
        return arg_keys + [da.key + '.ndarray']

    def __call__(self, *args, **kwargs):
        # get context from args
        distribution = self.determine_distribution(args, kwargs)
        context = distribution.context
        # push function
        self.push_fn(context, self.fn_key, self.fn)
        # vectorize the function
        exec_str = "%s = numpy.vectorize(%s)" % (self.fn_key, self.fn_key)
        context._execute(exec_str, targets=distribution.targets)

        # Find the first distarray, they should all be the same up to the data.
        for arg in args:
            if isinstance(arg, DistArray):
                # Create the output distarray.
                out = context.empty(arg.distribution, dtype=arg.dtype)
                # parse args
                args_str, kwargs_str = self.key_and_push_args(
                    args, kwargs, context=context,
                    da_handler=self.get_ndarray)

                # Call the function
                exec_str = ("if %s.ndarray.size != 0: %s.ndarray = "
                            "%s(*%s, **%s)")
                exec_str %= (out.key, out.key, self.fn_key, args_str,
                             kwargs_str)
                context._execute(exec_str, targets=distribution.targets)
                return out
