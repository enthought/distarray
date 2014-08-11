# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Utilities for running Distarray in MPI mode.
"""

from __future__ import absolute_import

import types

from mpi4py import MPI as mpi

from distarray.utils import uid
from distarray.localapi.proxyize import Proxy


client_rank = 0


def get_comm_world():
    return mpi.COMM_WORLD


def get_world_rank():
    return get_comm_world().rank


def push_function(context, key, func, targets=None):
    targets = targets or context.targets

    if not isinstance(func, types.BuiltinFunctionType):
        func_code = func.__code__
        func_globals = func.__globals__  # noqa
        func_name = func.__name__
        func_defaults = func.__defaults__
        func_closure = func.__closure__
        func_data = ('function', func_code, func_name, func_defaults, func_closure)
    else:
        func_data = ('builtin_function_or_method', func)

    def reassemble_and_store_func(key_dummy_container, func_data):
        import types
        from importlib import import_module
        from distarray.utils import set_from_dotted_name
        key = key_dummy_container[0]
        main = import_module('__main__')
        if func_data[0] == 'function':
            code, name, defaults, closure = func_data[1:]
            func = types.FunctionType(code=code, globals=main.__dict__,
                                      name=name, argdefs=defaults,
                                      closure=closure)
        elif func_data[0] == 'builtin_function_or_method':
            func = func_data[1]
        set_from_dotted_name(key, func)

    context.apply(reassemble_and_store_func, args=((key,), func_data),
                  targets=context.targets)


def get_nengines():
    """Get the number of engines which must be COMM_WORLD.size - 1 (for the
    client)
    """
    return get_comm_world().size - 1


def _set_on_main(name, obj):
    """Add obj as an attribute to the __main__ module with alias `name` like:
        __main__.name = obj
    """
    return Proxy(name, obj, '__main__')


def make_intercomm(targets=None):
    world = get_comm_world()
    world_rank = world.rank
    # create a comm that is split into client and engines.
    targets = targets or list(range(world.size - 1))

    if world_rank == client_rank:
        split_world = world.Split(0, 0)
    else:
        split_world = world.Split(1, world_rank)

    # create the intercomm
    if world_rank == client_rank:
        intercomm = split_world.Create_intercomm(0, world, 1)
    else:
        intercomm = split_world.Create_intercomm(0, world, 0)
    return intercomm


def make_base_comm():
    """
    Creates an intracomm consisting of all the engines. Then sets:
        `__main__._base_comm = comm_name`
    """
    world = get_comm_world()
    if world.rank == 0:
        comm_name = uid()
    else:
        comm_name = ''
    comm_name = world.bcast(comm_name)

    engines = world.group.Excl([client_rank])
    engine_comm = world.Create(engines)
    return _set_on_main(comm_name, engine_comm)


def make_targets_comm(targets):
    world = get_comm_world()
    world_rank = world.rank

    if len(targets) > world.size:
        raise ValueError("The number of engines (%s) is less than the number"
                         " of targets you want (%s)." % (world.size - 1,
                                                         len(targets)))
    targets = targets or list(range(world.size - 1))
    # get a universal name for the out comm
    if world_rank == 0:
        comm_name = uid()
    else:
        comm_name = ''
    comm_name = world.bcast(comm_name)

    # create a mapping from the targets to world ranks
    all_ranks = range(1, world.size)
    all_targets = range(world.size - 1)
    target_to_rank_map = {t: r for t, r in zip(all_targets, all_ranks)}

    # map the targets to the world ranks
    mapped_targets = [target_to_rank_map[t] for t in targets]

    # create the targets comm
    targets_group = world.group.Incl(mapped_targets)
    targets_comm = world.Create(targets_group)
    return _set_on_main(comm_name, targets_comm)


def setup_engine_comm(targets=None):
    # create a comm that is split into client and engines.
    world = get_comm_world()
    world_rank = world.rank
    targets = range(world.size - 1) if targets is None else targets
    name = uid()
    if world_rank == client_rank:
        split_world = world.Split(0, 0)
    elif (world_rank + 1) in targets:
        split_world = world.Split(1, world_rank)
        _set_on_main(name, split_world)
    else:
        world.Split(2, world_rank)


def initial_comm_setup():
    """Setup client and engine intracomm, and intercomm."""
    world = get_comm_world()
    world_rank = world.rank

    # create a comm that is split into client and engines.
    if world_rank == client_rank:
        split_world = world.Split(0, 0)
    else:
        split_world = world.Split(1, world_rank)

    # create the intercomm
    if world_rank == client_rank:
        intercomm = split_world.Create_intercomm(0, world, 1)
    else:
        intercomm = split_world.Create_intercomm(0, world, 0)

    return intercomm


def is_solo_mpi_process():
    if get_comm_world().size == 1:
        return True
    else:
        return False
