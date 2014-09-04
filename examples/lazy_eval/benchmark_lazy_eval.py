#!/usr/bin/env python
# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
See if lazy evaluation (promise pipelining) can be more efficient than eager
evaluation.

"""

from __future__ import print_function, division

from sys import stderr
from timeit import default_timer as time
from pprint import pprint

from distarray.globalapi import Context, tanh


nops_list = range(1, 10002, 1000)
arr_shape = (10, 10)
arr_size = arr_shape[0] * arr_shape[1]

context = Context()

eager = []
lazy = []

for nops in nops_list:

    arr = context.ones(arr_shape)
    start = time()
    for _ in range(nops):
        arr = tanh(arr)
    result = time() - start
    eager.append(result)
    print('.', end='', file=stderr, flush=True)

    arr = context.ones(arr_shape)
    start = time()
    with context.lazy_eval():
        for i in range(nops):
            arr = tanh(arr)
    result = time() - start
    lazy.append(result)
    print('.', end='', file=stderr, flush=True)

print(file=stderr, flush=True)
pprint({"nops": list(nops_list),
        "Eager": eager,
        "Lazy": lazy,
       }, stream=stderr)
