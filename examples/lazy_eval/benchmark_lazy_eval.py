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

import json
import datetime
from sys import stderr
from timeit import default_timer as time

import numpy
from distarray.globalapi import Context, tanh


def benchmark(nops_list, arr_shape):
    context = Context()

    data = {"nops": list(nops_list),
            "Numpy": [],
            "Eager": [],
            "Lazy": [],
            }

    total_tests = len(nops_list) * 3
    test_num = 1
    for nops in nops_list:

        # bench Numpy
        arr = numpy.ones(arr_shape)
        start = time()
        for _ in range(nops):
            arr = numpy.tanh(arr)
        result = time() - start
        data['Numpy'].append(result)
        print('({}/{}), Numpy, {} ops, {:0.3f} s'.format(test_num, total_tests, nops, result),
            file=stderr, flush=True)
        test_num += 1

        # bench DistArray eager eval
        arr = context.ones(arr_shape)
        start = time()
        for _ in range(nops):
            arr = tanh(arr)
        result = time() - start
        data['Eager'].append(result)
        print('({}/{}), Eager, {} ops, {:0.3f} s'.format(test_num, total_tests, nops, result),
            file=stderr, flush=True)
        test_num += 1

        # bench DistArray lazy eval
        arr = context.ones(arr_shape)
        start = time()
        with context.lazy_eval():
            for i in range(nops):
                arr = tanh(arr)
        result = time() - start
        data['Lazy'].append(result)
        print('({}/{}),  Lazy, {} ops, {:0.3f} s'.format(test_num, total_tests, nops, result),
            file=stderr, flush=True)
        test_num += 1

    return data


def save_data(data, note=''):
    now = datetime.datetime.now()
    filename = '_'.join((now.strftime("%Y-%m-%dT%H-%M-%S"), note)) + ".json"
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4)


def main(nops_list=None, arr_shape=None, note=''):
    nops_list = range(1, 20002, 1000) if nops_list is None else nops_list
    arr_shape = (10, 10) if arr_shape is None else arr_shape
    data = benchmark(nops_list, arr_shape)
    save_data(data, note)

    return data

if __name__ == '__main__':
    main()
