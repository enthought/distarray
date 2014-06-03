# encoding: utf-8
# -----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# -----------------------------------------------------------------------------

import importlib

from distarray.utils import DISTARRAY_BASE_NAME


class Proxyize(object):
    def __init__(self, context_key):
        self.context_key = context_key
        self.count = None
        self.state = None
        self.main = importlib.import_module('__main__')
        self.context = getattr(self.main, self.context_key)

    def set_state(self, state):
        self.state = state
        self.count = 0

    def str_counter(self):
        res = str(self.count)
        self.count += 1
        return res

    def next_name(self):
        if (self.state is None) or (self.count is None):
            raise RuntimeError("proxyize's state must be set before being "
                               "called.")
        return DISTARRAY_BASE_NAME + self.state + self.str_counter()

    def __call__(self, obj):
        new_name = self.next_name()
        setattr(self.main, new_name, obj)
        return new_name
