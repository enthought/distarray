"""
ODIN: ODin Isn't Numpy
"""

from itertools import chain
from functools import wraps

from IPython.parallel import Client
from distarray.client import Context, DistArray


# Set up a global Context on import
_global_client = Client()
_global_view = _global_client[:]
_global_context = Context(_global_view)
context = _global_context


def flatten(lst):
    """Given a list of lists, return a flattened list.

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
        return all(element == lst[0] for element in lst[1:])


def key_and_push_args(subcontext, arglist):
    """For each arg in arglist, get or generate a key (UUID).

    For DistArray objects, just get the existing key.  For
    everything else, generate a key and push the value to the engines

    Parameters
    ----------
    subcontext : Context
    arglist : List of objects to key and/or push

    Returns
    -------
    arg_keys : list of keys
    """

    arg_keys = []
    for arg in arglist:
        if isinstance(arg, DistArray):
            # if a DistArray, use its existing key
            arg_keys.append(arg.key)
            is_same_context = (subcontext == arg.context)
            err_msg_fmt = "DistArray context mismatch: {} {}"
            assert is_same_context, err_msg_fmt.format(subcontext, arg.context)
        else:
            # if not a DistArray, key it and push it to engines
            arg_keys.extend(subcontext._key_and_push(arg))
    return arg_keys


def determine_context(definition_context, args):
    """Determine the Context for a function.

    Parameters
    ----------
    definition_context: Context object
        The Context in which the function was defined.

    args : iterable
        List of objects to inspect for context.  Objects that aren't of
        type DistArray are skipped.

    Returns
    -------
    Context
        If all provided DistArray objects have the same context.

    Raises
    ------
    ValueError
        Raised if all DistArray objects don't have the same context.
    """
    contexts = [definition_context] + [arg.context for arg in args if
                                       isinstance(arg, DistArray)]
    if not all_equal(contexts):
        errmsg = "All DistArray objects must be defined with the same context used for the function: {}"
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
    A DistArray (if locally it's a DistArray), a None (if locally
    it's a None).

    Raises
    ------
    TypeError for any other type besides those handled above

    """
    type_key = subcontext._generate_key()
    type_statement = "{} = str(type({}))".format(type_key, result_key)
    subcontext._execute0(type_statement)
    result_type_str = subcontext._pull0(type_key)

    if (result_type_str == "<type 'NoneType'>" or  # Python 2
            result_type_str == "<class 'NoneType'>"):  # Python 3
        result = None
    elif result_type_str == "<class 'distarray.core.denselocalarray.DenseLocalArray'>":
        result = DistArray(result_key, subcontext)
    else:
        msg = ("Type is {}.  @local is not yet implemented for return types"
                "other than DistArray and NoneType").format(result_type_str)
        raise TypeError(msg)

    return result


def local(fn):
    """Decorator indicating a function is run locally on engines.

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

    @wraps(fn)
    def inner(*args, **kwargs):

        subcontext = determine_context(_global_context, flatten((args, kwargs.values())))

        # generate keys for each parameter
        # push to engines if not a DistArray
        arg_keys = key_and_push_args(subcontext, args)
        kwarg_names = kwargs.keys()
        kwarg_keys = key_and_push_args(subcontext, kwargs.values())

        # build up a python statement as a string
        args_fmt = ','.join(['{}'] * len(arg_keys))
        kwargs_fmt = ','.join(['{}={}'] * len(kwarg_keys))
        # remove empty strings before joining
        fmts = (fmt for fmt in (args_fmt, kwargs_fmt) if fmt)
        fnargs_fmt = ','.join(fmts)
        statement_fmt = ''.join(['{} = {}(', fnargs_fmt, ')'])
        replacement_values = ([result_key, func_key] + arg_keys +
                              flatten(zip(kwarg_names, kwarg_keys)))
        statement = statement_fmt.format(*replacement_values)

        # execute it locally
        subcontext._execute(statement)

        return process_return_value(subcontext, result_key)

    return inner
