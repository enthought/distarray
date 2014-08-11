# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Functions for cleaning up `DistArray` objects from IPython parallel engines.
"""

from __future__ import print_function, absolute_import

from distarray.globalapi.ipython_utils import IPythonClient


def cleanup(view, module_name, prefix):
    """ Delete Context object with the given name from the given module"""
    def _cleanup(module_name, prefix):
        from importlib import import_module
        ns = import_module(module_name)
        for name in vars(ns).copy():
            if name.startswith(prefix):
                delattr(ns, name)

    view.apply_sync(_cleanup, module_name, prefix)


def cleanup_all(module_name, prefix):
    """ Connects to all engines and runs ``cleanup()`` on them. """
    c = IPythonClient()
    if c is None:
        return
    try:
        v = c[:]
        cleanup(v, module_name, prefix)
    finally:
        c.close()


def get_local_keys(view, prefix):
    """ Returns a dictionary of keyname -> target_list mapping for all names
        that start with ``prefix`` on engines in ``view``.
    """

    def get_keys_engine(prefix):
        return [key for key in globals() if key.startswith(prefix)]

    keys_from_target = view.apply_async(get_keys_engine, prefix).get_dict()

    targets_from_key = {}
    for target, keys in keys_from_target.items():
        for key in keys:
            targets_from_key.setdefault(key, []).append(target)

    return targets_from_key

def clear(view):
    """ Removes all distarray-related modules from engines' sys.modules."""

    def clear_engine():
        from sys import modules
        orig_mods = set(modules)
        for m in modules.copy():
            if m.startswith('distarray'):
                del modules[m]
        return sorted(orig_mods - set(modules))

    return view.apply_async(clear_engine).get_dict()

def clear_all():
    c = IPythonClient()
    if c is None:
        return
    try:
        v = c[:]
        mods = clear(v)
    finally:
        c.close()
    return mods
