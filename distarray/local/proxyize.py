# encoding: utf-8
# -----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# -----------------------------------------------------------------------------

from importlib import import_module

from distarray.utils import DISTARRAY_BASE_NAME

class Proxy(object):

    def __init__(self, name, obj, module_name):
        self.name = name
        self.module_name = module_name
        self.type_str = str(type(obj))
        namespace = import_module(self.module_name)
        setattr(namespace, self.name, obj)

    def dereference(self):
        """ Callable only on the engines. """
        namespace = import_module(self.module_name)
        return getattr(namespace, self.name)

    def cleanup(self):
        namespace = import_module(self.module_name)
        delattr(namespace, self.name)
        self.name = self.module_name = self.type_str = None


class Proxyize(object):

    def __init__(self):
        self.count = None
        self.state = None

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
        return Proxy(new_name, obj, '__main__')
