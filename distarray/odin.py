"""
ODIN: ODin Isn't Numpy
"""

from IPython.parallel import Client
from distarray.client import DistArrayContext, DistArrayProxy
from operator import add


# Set up a global DistArrayContext on import
_global_client = Client()
_global_view = _global_client[:]
_global_context = DistArrayContext(_global_view)
context = _global_context


def flatten(lst):
    """ Given a list of lists, return a flattened list.

    Only flattens one level.  For example,

    >>> flatten(zip(['a', 'b', 'c'], [1, 2, 3]))
    ['a', 1, 'b', 2, 'c', 3]

    >>> flatten([[1, 2], [3, 4, 5], [[5], [6], [7]]])
    [1, 2, 3, 4, 5, [5], [6], [7]]
    """
    if len(lst) == 0:
        return []
    else:
        return list(reduce(add, lst))


def key_and_push_args(context, arglist):
    """ For each arg in arglist, get or generate a key (UUID).

    For DistArrayProxy objects, just get the existing key.  For
    everything else, generate a key and push the value to the engines

    Parameters
    ----------
    context : DistArrayContext
    arglist : List of objects to key and/or push

    Returns
    -------
    arg_keys : list of keys
    """

    arg_keys = []
    for arg in arglist:
        if isinstance(arg, DistArrayProxy):
            # if a DistArrayProxy, use its existing key
            arg_keys.append(arg.key)
            is_self = (context == arg.context)
            err_msg_fmt = "distarray context mismatch: {} {}"
            assert is_self, err_msg_fmt.format(context, arg.context)
        else:
            # if not a DistArrayProxy, key it and push it to engines
            arg_keys.extend(context._key_and_push(arg))
    return arg_keys


def local(fn):
    """ Decorator indicating a function is run locally on engines.

    Parameters
    ----------
    fn : function to wrap to run locally on engines

    Returns
    -------
    fn : function wrapped to run locally on engines
    """
    # we want @local functions to be able to call each other, so push
    # their `__name__` as their key
    func_key = fn.__name__
    _global_context._push({func_key: fn})
    result_key = _global_context._generate_key()

    def inner(*args, **kwargs):

        subcontext = kwargs.pop('context', None)
        if subcontext is None:
            subcontext = context

        # generate keys for each parameter
        # push to engines if not a DistArrayProxy
        arg_keys = key_and_push_args(subcontext, args)
        kwarg_names = kwargs.keys()
        kwarg_keys = key_and_push_args(subcontext, kwargs.values())

        # build up a python statement as a string
        args_fmt = ','.join(['{}'] * len(arg_keys))
        kwargs_fmt = ','.join(['{}={}'] * len(kwarg_keys))
        fnargs_fmt = ','.join([args_fmt, kwargs_fmt])
        statement_fmt = ''.join(['{} = {}(', fnargs_fmt, ')'])
        replacement_values = ([result_key, func_key] + arg_keys +
                              flatten(zip(kwarg_names, kwarg_keys)))
        statement = statement_fmt.format(*replacement_values)

        # execute it locally and return the result as a DistArrayProxy
        subcontext._execute(statement)
        return DistArrayProxy(result_key, subcontext)

    return inner
