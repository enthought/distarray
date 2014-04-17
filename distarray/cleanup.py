# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from distarray.ipython_utils import IPythonClient


def cleanup(view, prefix):
    """ Delete keys with prefix from client's engines. """
    # Delete keys only from this context.
    def engine_cleanup(prefix):
        glb = globals()
        global_keys = list(globals())
        for gk in global_keys:
            if gk.startswith(prefix):
                del glb[gk]
    view.apply_async(engine_cleanup, prefix)

def cleanup_all(prefix):
    """ Connects to all engines and runs ``cleanup()`` on them. """
    try:
        c = IPythonClient()
    except IOError:  # If we can't create a client, return silently.
        return
    try:
        v = c[:]
        cleanup(v, prefix)
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
    try:
        c = IPythonClient()
    except IOError:  # If we can't create a client, return silently.
        return
    try:
        v = c[:]
        mods = clear(v)
    finally:
        c.close()
    return mods
