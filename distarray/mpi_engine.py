# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
The engine_loop function and utilities necessary for it.
"""

from functools import reduce
from importlib import import_module
import types

from distarray.local import LocalArray
from distarray.local.proxyize import Proxy

from distarray.mpionly_utils import (initial_comm_setup, make_targets_comm,
                                     get_comm_world)


class Engine(object):
    _BASE_COMM = None
    _INTERCOMM = None

    def __init__(self):
        self.world = get_comm_world()
        self.world_ranks = list(range(self.world.size))

        # make engine and client comm
        self.client_rank = 0
        self.engine_ranks = [i for i in self.world_ranks if i !=
                             self.client_rank]

        # make engines intracomm (Context._base_comm):
        base_comm, intercomm = initial_comm_setup()
        self.__class__._BASE_COMM = base_comm
        self.__class__._INTERCOMM = intercomm
        assert self.world.rank != 0
        while True:
            msg = self._INTERCOMM.recv(source=0)
            val = self.parse_msg(msg)
            if val == 'kill':
                break

    def arg_kwarg_proxy_converter(self, args, kwargs):
        module = import_module('__main__')
        # convert args
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, module.Proxy):
                args[i] = a.dereference()
        args = tuple(args)

        # convert kwargs
        for k in kwargs.keys():
            val = kwargs[k]
            if isinstance(val, module.Proxy):
                kwargs[k] = val.dereference()

        return args, kwargs

    def is_engine(self):
        if self.world.rank != self.client_rank:
            return True
        else:
            return False

    def parse_msg(self, msg):
        to_do = msg[0]
        what = {'func_call': self.func_call,
                'execute': self.execute,
                'push': self.push,
                'pull': self.pull,
                'kill': self.kill,
                'delete': self.delete,
                'make_targets_comm': self.engine_make_targets_comm,
                'builtin_call': self.builtin_call}
        func = what[to_do]
        ret = func(msg)
        return ret

    def delete(self, msg):
        obj = msg[1]
        if isinstance(obj, Proxy):
            obj.cleanup()
        else:
            name = obj
            try:
                module = import_module('__main__')
                delattr(module, name)
            except AttributeError:
                pass

    def func_call(self, msg):

        func_data = msg[1]
        args = msg[2]
        kwargs = msg[3]
        nonce, context_key = msg[4]
        autoproxyize = msg[5]

        module = import_module('__main__')
        module.proxyize.set_state(nonce)

        args, kwargs = self.arg_kwarg_proxy_converter(args, kwargs)

        new_func_globals = module.__dict__  # add proper proxyize, context_key
        new_func_globals.update({'proxyize': module.proxyize,
                                'context_key': context_key})

        new_func = types.FunctionType(func_data[0], new_func_globals,
                                      func_data[1], func_data[2], func_data[3])

        res = new_func(*args, **kwargs)
        if autoproxyize and isinstance(res, LocalArray):
            res = module.proxyize(res)
        self._INTERCOMM.send(res, dest=self.client_rank)

    def execute(self, msg):
        main = import_module('__main__')
        code = msg[1]
        exec(code, main.__dict__)

    def push(self, msg):
        d = msg[1]
        module = import_module('__main__')
        for k, v in d.items():
            pieces = k.split('.')
            place = reduce(getattr, [module] + pieces[:-1])
            setattr(place, pieces[-1], v)

    def pull(self, msg):
        name = msg[1]
        module = import_module('__main__')
        res = reduce(getattr, [module] + name.split('.'))
        self._INTERCOMM.send(res, dest=self.client_rank)

    def kill(self, msg):
        """Break out of the engine loop."""
        return 'kill'

    def engine_make_targets_comm(self, msg):
        targets = msg[1]
        make_targets_comm(targets)

    def builtin_call(self, msg):
        func = msg[1]
        args = msg[2]
        kwargs = msg[3]

        args, kwargs = self.arg_kwarg_proxy_converter(args, kwargs)

        res = func(*args, **kwargs)
        self._INTERCOMM.send(res, dest=self.client_rank)
