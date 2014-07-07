# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import print_function

import numpy
from distarray.externals.six.moves import input

from distarray.dist import Context, Distribution
from distarray.dist.decorators import local
from pprint import pprint

context = Context()

numpy.set_printoptions(precision=2, linewidth=1000)


@local
def local_sin(da):
    """A simple @local function."""
    return numpy.sin(da)


@local
def local_sin_plus_50(da):
    """An @local function that calls another."""
    return local_sin(da) + 50


@local
def global_sum(da):
    """Reproducing the `sum` function in LocalArray."""
    from distarray.local.mpiutils import MPI
    from distarray.local import LocalArray
    from distarray.local.maps import Distribution

    local_sum = da.ndarray.sum()
    global_sum = da.distribution.comm.allreduce(local_sum, None, op=MPI.SUM)

    new_arr = numpy.array([global_sum])
    distribution = Distribution((1,), comm=da.comm)
    new_distarray = LocalArray(distribution, buf=new_arr)
    return new_distarray


if __name__ == '__main__':

    arr_len = 40

    print()
    input("Basic creation:")
    dist_b = Distribution(context, (arr_len,), dist={0: 'b'})
    dap_b = context.empty(dist_b)
    dist_c = Distribution(context, (arr_len,), dist={0: 'c'})
    dap_c = context.empty(dist_c)
    print("dap_b is a ", type(dap_b))
    print("dap_c is a ", type(dap_c))

    print()
    input("__setitem__:")
    for x in range(arr_len):
        dap_b[x] = x
        dap_c[x] = x
    pprint(dap_b.get_localarrays())
    pprint(dap_c.get_localarrays())

#    print
#    input("__getitem__ with slicing:")
#    print dap_b[19:34:2]
#    print dap_c[19:34:2]

    print()
    input("@local functions:")
    dap1 = local_sin(dap_b)
    pprint(dap1.get_localarrays())

    print()
    input("calling @local functions from each other:")
    dap2 = local_sin_plus_50(dap_b)
    pprint(dap2.get_localarrays())

    print()
    input("calling MPI from @local functions:")
    dap3 = global_sum(dap_b)
    pprint(dap3.get_localarrays())
