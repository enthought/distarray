# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import print_function

from distarray.ipython_utils import IPythonClient


def engine_cleanup(module_name, prefix):
    """ Remove variables with ``prefix`` prefix from the namespace of the
    module with ``module_name``. 
    """
    mod = __import__(module_name)
    ns = mod.__dict__
    keys = ns.keys()
    deleted = 0
    for k in keys:
        if k.startswith(prefix):
            del ns[k]
            deleted += 1
    count = sum([1 for k in ns if k.startswith(prefix)])
    return (deleted, count)


def cleanup(view, module_name, prefix):
    """ Delete keys with prefix from client's engines. """
    remaining = view.apply_async(engine_cleanup, module_name, prefix).get_dict()
    return remaining


def cleanup_all(module_name, prefix):
    """ Connects to all engines and runs ``cleanup()`` on them. """
    try:
        c = IPythonClient()
    except IOError:  # If we can't create a client, return silently.
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
