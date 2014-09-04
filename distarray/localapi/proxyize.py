# encoding: utf-8
# -----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# -----------------------------------------------------------------------------

from importlib import import_module

from distarray.utils import DISTARRAY_BASE_NAME, nonce


class LazyPlaceholder(object):
    pass


class Proxy(object):

    def __init__(self, name, obj, module_name, lazy=False):
        self.name = name
        self.module_name = module_name
        self.lazy = lazy
        self.type_str = str(type(obj))
        namespace = import_module(self.module_name)
        setattr(namespace, self.name, obj)

    def dereference(self):
        """Callable only on the engines."""
        namespace = import_module(self.module_name)
        return getattr(namespace, self.name)

    def cleanup(self):
        namespace = import_module(self.module_name)
        if 'lazy' not in self.name:
            delattr(namespace, self.name)
        self.name = self.module_name = self.type_str = None


def lazy_name():
    return DISTARRAY_BASE_NAME + "lazy_" + nonce()


def lazy_proxyize(name=None):
    """Return a Proxy object for a delayed ("lazy") value."""
    if name is None:
        name = lazy_name()
    return Proxy(name=name, obj=LazyPlaceholder(),
                 module_name='__main__', lazy=True)


class Proxyize(object):

    """Callable that, given an object, returns a Proxy object.

    You must call `set_state` on the instance before you can "call" it.
    """

    def __init__(self):
        self.count = None
        self.state = None

    def set_state(self, state):
        self.state = state
        self.count = 0

    def str_counter(self):
        """Return the str value of `self.count`, then increment its value."""
        res = str(self.count)
        self.count += 1
        return res

    def next_name(self):
        if (self.state is None) or (self.count is None):
            raise RuntimeError("proxyize's state must be set before being "
                               "called.")
        return DISTARRAY_BASE_NAME + self.state + self.str_counter()

    def __call__(self, obj):
        """Return a `Proxy` object given an object `obj`."""
        new_name = self.next_name()
        return Proxy(new_name, obj, '__main__')
