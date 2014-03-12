# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from __future__ import print_function

import numpy
import distarray
from distarray.externals.six.moves import input
from distarray import Context
from distarray.decorators import local
from pprint import pprint

context = Context()

numpy.set_printoptions(precision=2, linewidth=1000)
context.view.execute("import numpy")


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
    """Reproducing the `sum` function in densedistarray."""
    from distarray.mpiutils import MPI
    local_sum = da.local_array.sum()
    global_sum = da.comm.allreduce(local_sum, None, op=MPI.SUM)

    new_arr = numpy.array([global_sum])
    new_distarray = distarray.local.LocalArray((1,), buf=new_arr)
    return new_distarray


if __name__ == '__main__':

    arr_len = 40

    print()
    input("Basic creation:")
    dap_b = context.empty((arr_len,), dist={0: 'b'})
    dap_c = context.empty((arr_len,), dist={0: 'c'})
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
