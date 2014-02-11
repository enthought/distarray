from __future__ import print_function
import numpy
import distarray
from six.moves import input
from distarray import odin
from pprint import pprint


numpy.set_printoptions(precision=2, linewidth=1000)
odin.context.view.execute("import numpy")


@odin.remote
def remote_sin(da):
    """A simple @remote function."""
    return numpy.sin(da)


@odin.remote
def remote_sin_plus_50(da):
    """An @remote function that calls another."""
    return remote_sin(da) + 50


@odin.remote
def global_sum(da):
    """Reproducing the `sum` function in densedistarray."""
    from distarray.mpiutils import MPI
    remote_sum = da.remote_array.sum()
    global_sum = da.comm.allreduce(remote_sum, None, op=MPI.SUM)

    new_arr = numpy.array([global_sum])
    new_distarray = distarray.RemoteArray((1,), buf=new_arr)
    return new_distarray


if __name__ == '__main__':

    arr_len = 40

    print()
    input("Basic creation:")
    dap_b = odin.context.empty((arr_len,), dist={0: 'b'})
    dap_c = odin.context.empty((arr_len,), dist={0: 'c'})
    print("dap_b is a ", type(dap_b))
    print("dap_c is a ", type(dap_c))

    print()
    input("__setitem__:")
    for x in range(arr_len):
        dap_b[x] = x
        dap_c[x] = x
    pprint(dap_b.get_remotearrays())
    pprint(dap_c.get_remotearrays())

#    print
#    input("__getitem__ with slicing:")
#    print dap_b[19:34:2]
#    print dap_c[19:34:2]

    print()
    input("@remote functions:")
    dap1 = remote_sin(dap_b)
    pprint(dap1.get_remotearrays())

    print()
    input("calling @remote functions from each other:")
    dap2 = remote_sin_plus_50(dap_b)
    pprint(dap2.get_remotearrays())

    print()
    input("calling MPI from @remote functions:")
    dap3 = global_sum(dap_b)
    pprint(dap3.get_remotearrays())
