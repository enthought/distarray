"""
ODIN: ODin Isn't Numpy
"""

from itertools import chain

from IPython.parallel import Client
from distarray.client import DistArrayContext, DistArrayProxy


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
    return list(chain.from_iterable(lst))


def all_equal(lst):
    """Return True if all elements in `lst` are equal.

    Also returns True if list is empty.
    """
    if len(lst) == 0 or len(lst) == 1:
        return True  # vacuously True
    else:
        return all([element == lst[0] for element in lst[1:]])


def key_and_push_args(subcontext, arglist):
    """ For each arg in arglist, get or generate a key (UUID).

    For DistArrayProxy objects, just get the existing key.  For
    everything else, generate a key and push the value to the engines

    Parameters
    ----------
    subcontext : DistArrayContext
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
            is_same_context = (subcontext == arg.context)
            err_msg_fmt = "DistArrayProxy context mismatch: {} {}"
            assert is_same_context, err_msg_fmt.format(subcontext, arg.context)
        else:
            # if not a DistArrayProxy, key it and push it to engines
            arg_keys.extend(subcontext._key_and_push(arg))
    return arg_keys


def determine_context(args):
    """Determine the DistArrayContext for a function.

    Parameters
    ----------
    args : iterable
        List of objects to inspect for context.  Objects that aren't of
        type DistArrayProxy are skipped.

    Returns
    -------
    DistArrayContext
        If all provided DistArrayProxy objects have the same context.

    Raises
    ------
    ValueError
        Raised if all DistArrayProxy objects don't have the same context.
    """
    contexts = [arg.context for arg in args if isinstance(arg, DistArrayProxy)]
    if len(contexts) == 0:
        return context  # use the module-provided context
    elif not all_equal(contexts):
        errmsg = ("All DistArrayProxy objects must be defined "
                  "in the same context: {}")
        raise ValueError(errmsg.format(contexts))
    else:
        return contexts[0]


def process_return_value(subcontext, result_key):
    """Figure out what to return on the Client.

    Parameters
    ----------
    key : string
        Key corresponding to wrapped function's return value.

    Returns
    -------
    A DistArrayProxy (if locally it's a DistArray), a None (if locally
    it's a None).

    Raises
    ------
    TypeError for any other type besides those handled above

    """
    type_key = subcontext._generate_key()
    type_statement = "{} = str(type({}))".format(type_key, result_key)
    subcontext._execute0(type_statement)
    result_type_str = subcontext._pull0(type_key)

    if result_type_str == "<type 'NoneType'>":
        result = None
    elif result_type_str == "<class 'distarray.core.densedistarray.DenseDistArray'>":
        result = DistArrayProxy(result_key, subcontext)
    else:
        msg = ("@local not yet implemented for return types other "
               "than DistArray and NoneType")
        raise TypeError(msg)

    return result


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

        subcontext = determine_context(flatten((args, kwargs.values())))

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

        return process_return_value(subcontext, result_key)

    return inner
