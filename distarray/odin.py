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

        func_key = context._generate_key()
        context.view.push({func_key: fn}, targets=context.targets,
                          block=True)
        result_key = context._generate_key()

        def inner(a):
            err_msg_fmt = "distarray context mismatch: {} {}"
            assert context == a.context, err_msg_fmt.format(context, a.context)
            context._execute('%s = %s(%s)' % (result_key, func_key, a.key))
            return DistArrayProxy(result_key, context)

        return inner

    return wrap
