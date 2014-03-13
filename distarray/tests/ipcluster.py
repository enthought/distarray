# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

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


def start(n=4):
    """Convenient way to start an ipcluster for testing.

    Doesn't exit until the ipcluster prints a success message.
    """
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


def stop():
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


def restart():
    """Convenient way to restart an ipcluster."""
    stop()

    started = False
    while not started:
        sleep(2)
        try:
            start()
        except RuntimeError:
            pass
        else:
            started = True


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
