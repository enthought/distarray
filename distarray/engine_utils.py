"""
The engine_loop function and utilities necessary for it.
"""
from functools import reduce
from importlib import import_module
import types

from mpi4py import MPI as mpi


import distarray
from distarray.utils import DISTARRAY_BASE_NAME as prefix

from distarray.dist.mpionly_utils import initial_comm_setup, make_targets_comm


world = mpi.COMM_WORLD
world_ranks = list(range(world.size))

# make engine and client comm
client_rank = 0
engine_ranks = [i for i in world_ranks if i != client_rank]


def is_engine():
    if world.rank != client_rank:
        return True
    else:
        return False


def parse_msg(msg):
    to_do = msg[0]
    what = {'func_call': func_call,
            'execute': execute,
            'push': push,
            'pull': pull,
            'kill': kill,
            'make_targets_comm': engine_make_targets_comm}
    func = what[to_do]
    ret = func(msg)
    return ret


def func_call(msg):

    func_data = msg[1]
    args = msg[2]
    kwargs = msg[3]
    nonce, context_key = msg[4]

    module = import_module('__main__')
    module.proxyize.set_state(nonce)

    # convert args
    args = list(args)
    for i, a in enumerate(args):
        if (isinstance(a, str) and a.startswith(prefix)):
            args[i] = reduce(getattr, [module] + a.split('.'))
    args = tuple(args)

    # convert kwargs
    for k in kwargs.keys():
        val = kwargs[k]
        if (isinstance(val, str) and val.startswith(prefix)):
            kwargs[k] = module.reduce(getattr, [module] + val.split('.'))

    new_func_globals = module.__dict__  # add proper proxyize, context_key
    new_func_globals.update({'proxyize': module.proxyize,
                             'context_key': context_key})

    new_func = types.FunctionType(func_data[0], new_func_globals, func_data[1],
                                  func_data[2], func_data[3])

    res = new_func(*args, **kwargs)
    distarray.INTERCOMM.send(res, dest=client_rank)


def execute(msg):
    main = import_module('__main__')
    code = msg[1]
    exec(code, main.__dict__)


def push(msg):
    d = msg[1]
    module = import_module('__main__')
    for k, v in d.items():
        pieces = k.split('.')
        place = reduce(getattr, [module] + pieces[:-1])
        setattr(place, pieces[-1], v)


def pull(msg):
    name = msg[1]
    module = import_module(__name__)
    res = getattr(module, name)
    world.send(res, dest=client_rank)


def kill(msg):
    """Break out of the engine loop."""
    return 'kill'


def engine_make_targets_comm(msg):
    targets = msg[1]
    make_targets_comm(targets)


def engine_loop():
    # make engines intracomm (Context._base_comm):
    initial_comm_setup()
    assert mpi.COMM_WORLD.rank != 0
    while True:
        msg = distarray.INTERCOMM.recv(source=0)
        val = parse_msg(msg)
        if val == 'kill':
            break
