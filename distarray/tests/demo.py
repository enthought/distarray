import numpy as np
import distarray
from distarray import odin

np.set_printoptions(precision=2, linewidth=1000)


@odin.local
def local_sin(da):
    """A simple @local function."""
    return np.sin(da)


@odin.local
def local_sin_plus_50(da):
    """An @local function that calls another."""
    return local_sin(da) + 50


@odin.local
def global_sum(da):
    """Reproducing the `sum` function in densedistarray."""
    from distarray.mpi.mpibase import MPI
    local_sum = da.local_array.sum()
    global_sum = da.comm.allreduce(local_sum, None, op=MPI.SUM)

    new_arr = np.array([global_sum])
    new_distarray = distarray.DistArray((1,), buf=new_arr)
    return new_distarray


if __name__ == '__main__':
    arr_len = 128

    print
    raw_input("Basic creation:")
    dap0 = odin.context.empty((arr_len,))
    print "dap0 is a ", type(dap0)

    print
    raw_input("__setitem__:")
    for x in xrange(arr_len):
        dap0[x] = x
    print dap0.get_localarrays()

    print
    raw_input("__getitem__ with slicing:")
    print dap0[30:40]

    print
    raw_input("@local functions:")
    dap1 = local_sin(dap0)
    print dap1.get_localarrays()

    print
    raw_input("calling @local functions from each other:")
    dap2 = local_sin_plus_50(dap0)
    print dap2.get_localarrays()

    print
    raw_input("calling MPI from @local functions:")
    dap3 = global_sum(dap0)
    print dap3.get_localarrays()
