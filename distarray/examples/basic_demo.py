# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import print_function

import numpy
from distarray.externals.six.moves import input

from distarray.dist import Context, Distribution
from pprint import pprint

context = Context()

numpy.set_printoptions(precision=2, linewidth=1000)


def local_sin(da):
    """A simple registered function."""
    return numpy.sin(da)
context.register(local_sin)

def local_sin_plus_50(da):
    """A registered function that calls another."""
    return local_sin(da) + 50
context.register(local_sin_plus_50)

def global_sum(da):
    """Reproducing the `sum` function in LocalArray."""
    from distarray.local.mpiutils import MPI
    from distarray.local import LocalArray
    from distarray.local.maps import Distribution

    local_sum = da.ndarray.sum()
    global_sum = da.comm.allreduce(local_sum, None, op=MPI.SUM)
    global_shape = (da.comm.Get_size(),)

    new_arr = numpy.array([global_sum])
    distribution = Distribution.from_shape(da.comm, global_shape,
                                           dist=('b',),
                                           grid_shape=global_shape)
    new_distarray = LocalArray(distribution, buf=new_arr)
    return new_distarray
context.register(global_sum)


if __name__ == '__main__':

    arr_len = 40

    print()
    input("Basic creation:")
    dist_b = Distribution(context, (arr_len,), dist={0: 'b'})
    dap_b = context.zeros(dist_b) + numpy.pi
    dist_c = Distribution(context, (arr_len,), dist={0: 'c'})
    dap_c = context.zeros(dist_c) + (2 * numpy.pi)
    print("dap_b is a ", type(dap_b))
    print("dap_c is a ", type(dap_c))

    print()
    input("__setitem__:")
    dap_b[:] = range(arr_len)
    for x in range(arr_len):
        dap_c[x] = x
    pprint(dap_b.tondarray())
    pprint(dap_c.tondarray())

    print()
    input("registered functions:")
    dap1 = context.local_sin(dap_b)
    pprint(dap1.get_localarrays())

    print()
    input("calling registered functions from each other:")
    dap2 = context.local_sin_plus_50(dap_b)
    pprint(dap2.get_localarrays())

    print()
    input("calling MPI from registered functions:")
    dap3 = context.global_sum(dap_b)
    pprint(dap3.tondarray())
