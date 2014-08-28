# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
The MPI-based `Engine` class.
"""

from collections import OrderedDict
from functools import reduce
from importlib import import_module
import types

from distarray.externals.six import next
from distarray.localapi import LocalArray
from distarray.localapi.proxyize import Proxy

from distarray.mpionly_utils import (initial_comm_setup,
                                     make_targets_comm,
                                     get_comm_world)


class Engine(object):

    """MPI-based worker class."""

    INTERCOMM = None

    def __init__(self):
        """Setup and execute the main ``recv`` loop."""
        self.world = get_comm_world()
        self.world_ranks = list(range(self.world.size))

        # make engine and client comm
        self.client_rank = 0
        self.engine_ranks = [i for i in self.world_ranks if i !=
                             self.client_rank]

        # make engines intracomm (Context._base_comm):
        Engine.INTERCOMM = initial_comm_setup()
        assert self.is_engine()
        self.lazy = False
        self._client_sendq = []
        self._value_from_name = OrderedDict()
        while True:
            msg = Engine.INTERCOMM.recv(source=self.client_rank)
            val = self.parse_msg(msg)
            if val == 'kill':
                break
        Engine.INTERCOMM.Free()

    def arg_kwarg_proxy_converter(self, args, kwargs):
        """Dereference proxy arguments and update computed lazy proxies."""
        module = import_module('__main__')
        # convert args
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, module.Proxy):
                args[i] = self._value_from_name.get(a.name, a).dereference()
        args = tuple(args)

        # convert kwargs
        for k in kwargs.keys():
            val = kwargs[k]
            if isinstance(val, module.Proxy):
                kwargs[k] = val.dereference()

        return args, kwargs

    def is_engine(self):
        """Is this an engine (as opposed to the client)?"""
        if self.world.rank != self.client_rank:
            return True
        else:
            return False

    def parse_msg(self, msg):
        """Given a message, execute the proper function."""
        to_do = msg[0]
        what = {'func_call': self.func_call,
                'execute': self.execute,
                'push': self.push,
                'pull': self.pull,
                'kill': self.kill,
                'free_comm': self.free_comm,
                'delete': self.delete,
                'make_targets_comm': self.engine_make_targets_comm,
                'builtin_call': self.builtin_call,
                'process_message_queue': self.process_message_queue}
        func = what[to_do]
        ret = func(msg)
        return ret

    def delete(self, msg):
        """Process the 'delete' message.

        Cleans up the namespace.
        """
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
        """Process the 'func_call' message."""
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
        self._client_send(res)

    def execute(self, msg):
        """Process the 'execute' message."""
        main = import_module('__main__')
        code = msg[1]
        exec(code, main.__dict__)

    def push(self, msg):
        """Process the 'push' message."""
        d = msg[1]
        module = import_module('__main__')
        for k, v in d.items():
            pieces = k.split('.')
            place = reduce(getattr, [module] + pieces[:-1])
            setattr(place, pieces[-1], v)

    def pull(self, msg):
        """Process the 'pull' message."""
        name = msg[1]
        module = import_module('__main__')
        res = reduce(getattr, [module] + name.split('.'))
        self._client_send(res)

    def free_comm(self, msg):
        """Call `Free` on a communicator."""
        comm = msg[1].dereference()
        comm.Free()

    def kill(self, msg):
        """Process the 'kill' message.

        Breaks out of the engine loop.
        """
        return 'kill'

    def engine_make_targets_comm(self, msg):
        """Process the 'make_targets_comm' message."""
        targets = msg[1]
        make_targets_comm(targets)

    def builtin_call(self, msg):
        """Process the 'builtin_call' message."""
        func = msg[1]
        args = msg[2]
        kwargs = msg[3]

        args, kwargs = self.arg_kwarg_proxy_converter(args, kwargs)

        res = func(*args, **kwargs)
        self._client_send(res)

    def process_message_queue(self, msg):
        """Process the 'process_message_queue' message.

        This temporarily puts the engine in ``lazy`` mode, queueing up all the
        replies (`_client_send`) to be sent at once.
        """
        # we need the recvq (msg[1]) to see which values are returned from
        # which expression
        lazy_proxies = msg[1]
        # set up mapping from lazy_proxy names to their computed values
        # values to be filled in as the queue is processed
        self._value_from_name = OrderedDict([(lp.name, None) for lp in
                                             lazy_proxies])
        self._current_rval = iter(self._value_from_name)
        msgq = msg[2]
        self.lazy = True  # 'process_message_queue' only received in lazy mode
        for submsg in msgq:
            val = self.parse_msg(submsg)
            if val == 'kill':
                return val
        self.lazy = False
        self._client_send(self._client_sendq)
        self._client_sendq = []
        self._value_from_name = OrderedDict()

    def _client_send(self, msg):
        """Send a message to the client.

        If in lazy mode, just queue up the message.
        """
        if self.lazy:
            self._value_from_name[next(self._current_rval)] = msg
            self._client_sendq.append(msg)
        else:
            Engine.INTERCOMM.send(msg, dest=self.client_rank)
