"""
ODIN: ODin Isn't Numpy
"""

from distarray.client import DistArrayProxy


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
            keys = []
            for arg in args:
                if isinstance(arg, DistArrayProxy):
                    # if a DistArrayProxy, use its existing key
                    keys.append(arg.key)
                    is_self = (context == arg.context)
                    err_msg_fmt = "distarray context mismatch: {} {}"
                    assert is_self, err_msg_fmt.format(context, arg.context)
                else:
                    # if not a DistArrayProxy, key it and push it to engines
                    keys.extend(context._key_and_push(arg))

            # build up a python statement as a string
            # then execute it, returning the result as a DistArrayProxy
            args_fmt = ','.join(['{}'] * len(keys))
            statement_fmt = '{} = {}(' + args_fmt + ')'
            replacement_values = [result_key, func_key] + keys
            statement = statement_fmt.format(*replacement_values)
            context._execute(statement)
            return DistArrayProxy(result_key, context)

        return inner

    return wrap
