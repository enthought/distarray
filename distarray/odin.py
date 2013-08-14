"""
ODIN: ODin Isn't Numpy
"""

from distarray.client import DistArrayProxy
from operator import add


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


def key_and_push_args(context, args):
    arg_keys = []
    for arg in args:
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


def key_and_push_kwargs(context, kwargs):
    kwarg_names = []
    kwarg_keys = []
    for kwarg_name, kwarg_value in kwargs.items():
        kwarg_names.append(kwarg_name)
        if isinstance(kwarg_value, DistArrayProxy):
            # if a DistArrayProxy, use its existing key
            kwarg_keys.append(kwarg_value.key)
            is_self = (context == kwarg_value.context)
            err_msg_fmt = "distarray context mismatch: {} {}"
            assert is_self, err_msg_fmt.format(context, kwarg_value.context)
        else:
            # if not a DistArrayProxy, key it and push it to engines
            kwarg_keys.extend(context._key_and_push(kwarg_value))
    return kwarg_names, kwarg_keys


def local(context):
    """ Decorator indicating a function is run locally on engines.

    Parameters
    ----------
    context : DistArrayContext

    Returns
    -------
    fn : function wrapped to run locally on engines
    """

    def wrap(fn):

        func_key = context._key_and_push(fn)[0]
        result_key = context._generate_key()

        def inner(*args, **kwargs):
            # generate keys for each parameter
            # push to engines if not a DistArrayProxy
            arg_keys = key_and_push_args(context, args)
            kwarg_names, kwarg_keys = key_and_push_kwargs(context, kwargs)

            # build up a python statement as a string
            args_fmt = ','.join(['{}'] * len(arg_keys))
            kwargs_fmt = ','.join(['{}={}'] * len(kwarg_keys))
            statement_fmt = '{} = {}(' + args_fmt + ',' + kwargs_fmt + ')'
            replacement_values = ([result_key, func_key] + arg_keys +
                                  flatten(zip(kwarg_names, kwarg_keys)))
            statement = statement_fmt.format(*replacement_values)

            # execute it locally and return the result as a DistArrayProxy
            context._execute(statement)
            return DistArrayProxy(result_key, context)

        return inner

    return wrap
