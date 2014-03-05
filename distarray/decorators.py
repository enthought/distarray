"""
Decorators
"""
import functools

from distarray.client import DistArray
from distarray.context import Context
from distarray.error import ContextError
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
        context._push({fn_key: fn})

    def determine_context(self, args, kwargs):
        """ Determine a context from a functions arguments."""

        contexts = []
        # inspect args for a context
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, DistArray):
                contexts.append(arg.context)
            elif isinstance(arg, Context):
                contexts.append(arg)

        # check the args had a context
        if contexts == []:
            raise TypeError('Function must take DistArray or Context objects.')

        # check that all contexts are equal
        if not contexts.count(contexts[0]) == len(contexts):
            msg = ("Arguments must use the same Context (given arguments of "
                   "type %r)")
            msg %= (tuple(set(contexts)),)
            raise ContextError(msg)

        return contexts[0]

    def key_and_push_args(self, args, kwargs, context=None, da_handler=None):
        """
        Push a tuple of args and dict of kwargs to the engines. Return a
        tuple with keys corresponding to args values on the engines. And a
        dictionary with the same keys and values which are the keys to the
        input dictionary's values.

        This allows us to use the following interface to execute code on
        the engines:

        def foo(*args, **kwargs):
            args, kwargs = _key_and_push_args(args, kwargs)
            exec_str = "remote_foo(*%s, **%s)"
            exec_str %= (args, kwargs)
            context.execute(exec_str)
        """

        if context is None:
            context = self.determine_context(args, kwargs)

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
        context._push(push_keys)

        # build arg string
        arg_str = '(' + ', '.join(arg_keys) + ',)'

        # build kwarg string
        kwarg_iter = ["'%s': %s" % (k, v) for (k, v) in kwargs.items()]
        kwarg_str = '{' + ', '.join(kwarg_iter) + '}'

        return arg_str, kwarg_str

    def process_return_value(self, context, result_key):
        """Figure out what to return on the Client.

        Parameters
        ----------
        key : string
            Key corresponding to wrapped function's return value.

        Returns
        -------
        A DistArray (if locally all values are DistArray), a None (if
        locally all values are None), or else, pull the result back to the
        client and return it.  If all but one of the pulled values is None,
        return that non-None value only.
        """
        type_key = context._generate_key()
        type_statement = "{} = str(type({}))".format(type_key, result_key)
        context._execute(type_statement)
        result_type_str = context._pull(type_key)

        def is_NoneType(typestring):
            return (typestring == "<type 'NoneType'>" or
                    typestring == "<class 'NoneType'>")

        def is_LocalArray(typestring):
            return (typestring == "<class 'distarray.local.denselocalarray"
                                  ".DenseLocalArray'>")

        if all(is_LocalArray(r) for r in result_type_str):
            result = DistArray(result_key, context)
        elif all(is_NoneType(r) for r in result_type_str):
            result = None
        else:
            result = context._pull(result_key)
            if has_exactly_one(result):
                result = next(x for x in result if x is not None)

        return result


class local(DecoratorBase):
    """Decorator to run a function locally on the engines."""

    def __call__(self, *args, **kwargs):
        # get the context
        if self.context is None:
            # get context from args
            self.context = self.determine_context(args, kwargs)
            # push function
            self.push_fn(self.context, self.fn_key, self.fn)

        args, kwargs = self.key_and_push_args(args, kwargs,
                                              context=self.context)
        result_key = self.context._generate_key()

        exec_str = "%s = %s(*%s, **%s)"
        exec_str %= (result_key, self.fn_key, args, kwargs)
        self.context._execute(exec_str)

        return self.process_return_value(self.context, result_key)


class vectorize(DecoratorBase):
    """
    Analogous to numpy.vectorize. Input DistArray's must all be the
    same shape, and this will be the shape of the output distarray.
    """

    def get_local_array(self, da, arg_keys):
        return arg_keys + [da.key + '.local_array']

    def __call__(self, *args, **kwargs):
        if self.context is None:
            # get context from args
            self.context = self.determine_context(args, kwargs)
            # push function
            self.push_fn(self.context, self.fn_key, self.fn)
        # vectorize the function
        exec_str = "%s = numpy.vectorize(%s)" % (self.fn_key, self.fn_key)
        self.context._execute(exec_str)

        # Find the first distarray, they should all be the same up to the data.
        for arg in args:
            if isinstance(arg, DistArray):
                # Create the output distarray.
                out = self.context.empty(arg.shape, dtype=arg.dtype,
                                         dist=arg.dist,
                                         grid_shape=arg.grid_shape)
                # parse args
                args_str, kwargs_str = self.key_and_push_args(
                    args, kwargs, context=self.context,
                    da_handler=self.get_local_array)

                # Call the function
                exec_str = ("if %s.local_array.size != 0: %s.local_array = "
                            "%s(*%s, **%s)")
                exec_str %= (out.key, out.key, self.fn_key, args_str,
                             kwargs_str)
                self.context._execute(exec_str)
                return out
