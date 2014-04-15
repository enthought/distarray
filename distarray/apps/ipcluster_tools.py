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
from distarray.context import Context
from time import sleep
from subprocess import Popen, PIPE


if six.PY2:
    ipcluster_cmd = 'ipcluster'
elif six.PY3:
    ipcluster_cmd = 'ipcluster3'
else:
    raise NotImplementedError("Not run with Python 2 *or* 3?")


def start(n=4, engines=None, **kwargs):
    """Convenient way to start an ipcluster for testing.

    Doesn't exit until the ipcluster prints a success message.
    """
    if engines is None:
        engines = "--engines=MPIEngineSetLauncher"

    cluster = Popen([ipcluster_cmd, 'start', '-n', str(n), engines],
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


def stop(**kwargs):
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


def restart(n=4, engines=None, **kwargs):
    """Convenient way to restart an ipcluster."""
    stop()

    started = False
    while not started:
        sleep(2)
        try:
            start(n=n, engines=engines)
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


def clear(**kwargs):
    from IPython.parallel import Client
    c = Client()
    dv = c[:]
    dv.execute(_RESET_ENGINE_DISTARRAY, block=True)
    mods = dv['deleted_mods']
    print("The following modules were removed from the engines' namespaces:")
    for mod in mods[0]:
        print('    ' + mod)
    dv.clear()


def dump(**kwargs):
    """ Print out key names that exist on the engines. """
    context = Context()
    keylist = context.dump_keys(all_other_contexts=True)
    num_keys = len(keylist)
    print('*** %d ENGINE KEYS ***' % (num_keys))
    for key, targets in keylist:
        print('%s : %r' % (key, targets))


def purge(**kwargs):
    """ Remove keys from the engine namespaces. """
    print('Purging keys from engines...')
    context = Context()
    context.cleanup(all_other_contexts=True)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd not in 'start stop restart reset'.split():
        sys.exit("Error: %r not a valid command." % cmd)
    globals()[cmd]()
