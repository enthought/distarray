# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Functions for starting and stopping ipclusters.
"""

from __future__ import print_function

import sys
from distarray.externals import six
from time import sleep
from subprocess import Popen, PIPE


if six.PY2:
    ipcluster_cmd = 'ipcluster'
elif six.PY3:
    ipcluster_cmd = 'ipcluster3'
else:
    raise NotImplementedError("Not run with Python 2 *or* 3?")


def start(args):
    """Convenient way to start an ipcluster for testing.

    Doesn't exit until the ipcluster prints a success message.
    """
    nengines = args.nengines
    engines = "--engines=MPIEngineSetLauncher"
    cluster = Popen([ipcluster_cmd, 'start', '-n', str(nengines), engines],
                    stdout=PIPE, stderr=PIPE)

    started = "Engines appear to have started successfully"
    running = "CRITICAL | Cluster is already running with"
    while True:
        line = cluster.stderr.readline().decode()
        if not line:
            break
        print(line, end='')
        if (started in line):
            break
        elif (running in line):
            raise RuntimeError("ipcluster is already running.")


def stop(args):
    """Convenient way to stop an ipcluster."""
    stopping = Popen([ipcluster_cmd, 'stop'], stdout=PIPE, stderr=PIPE)

    stopped = "Stopping cluster"
    not_running = ("CRITICAL | Could not read pid file, cluster "
                   "is probably not running.")
    while True:
        line = stopping.stderr.readline().decode()
        if not line:
            break
        print(line, end='')
        if (stopped in line) or (not_running in line):
            break


def restart(args):
    """Convenient way to restart an ipcluster."""
    stop(args)

    started = False
    while not started:
        sleep(2)
        try:
            start(args)
        except RuntimeError:
            pass
        else:
            started = True


_RESET_ENGINE_DISTARRAY = '''
from sys import modules
orig_mods = set(modules)
for m in modules.copy():
    if m.startswith('distarray'):
        del modules[m]
deleted_mods = sorted(orig_mods - set(modules))
'''

def reset(args):
    from IPython.parallel import Client
    c = Client()
    dv = c[:]
    dv.execute(_RESET_ENGINE_DISTARRAY, block=True)
    mods = dv['deleted_mods']
    print("The following modules were removed from the engines' namespaces:")
    for mod in mods[0]:
        print('    ' + mod)
    dv.clear()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_start = subparsers.add_parser('start')
    parser_start.add_argument('nengines', type=int)
    parser_start.set_defaults(func=start)

    parser_restart = subparsers.add_parser('restart')
    parser_restart.add_argument('nengines', type=int)
    parser_restart.set_defaults(func=restart)

    subparsers.add_parser('stop').set_defaults(func=stop)
    subparsers.add_parser('reset').set_defaults(func=reset)

    args = parser.parse_args()
    args.func(args)
