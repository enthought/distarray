import numpy as np
from distarray.client import DistArrayContext, DistArrayProxy
from IPython.parallel import Client

c = Client()
dv = c[:]
dac = DistArrayContext(dv)

da = dac.empty((1024, 1024))
da.fill(2 * np.pi)


def local(context):

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


@local(dac)
def localsin(da):
    return np.sin(da)


@local(dac)
def localadd50(da):
    return da + 50


@local(dac)
def localsum(da):
    return np.sum(da)


dv.execute('import numpy as np')
db = localsin(da)
dc = localadd50(da)
dd = localsum(da)
#assert_allclose(db, 0)
