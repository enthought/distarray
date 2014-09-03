#!/usr/bin/env python
# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
See if lazy evaluation (promise pipelining) can be more efficeint than eager
evaluation.

"""

from distarray.globalapi import Context, tanh
from timeit import default_timer as time

nops = 10000
arr_shape = (100, 100)

context = Context()

arr = context.ones(arr_shape)
start = time()
for _ in range(nops):
    arr = tanh(arr)
print("Eager time: {:0.2f}".format(time() - start))

arr = context.ones(arr_shape)
start = time()
with context.lazy_eval():
    for i in range(nops):
        arr = tanh(arr)
print("Lazy time: {:0.2f}".format(time() - start))
