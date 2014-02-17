"""
Simple runner for `ipcluster start` or `ipcluster stop` on Python 2 or 3, as
appropriate.
"""

from __future__ import print_function

import sys
import six
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
        line = cluster.stderr.readline().strip().decode()
        print(line)
        if (started in line) or (running in line):
            break


def stop():
    """Convenient way to stop an ipcluster."""
    stopping = Popen([ipcluster_cmd, 'stop'], stdout=PIPE, stderr=PIPE)

    stopped = "Stopping cluster"
    not_running = ("CRITICAL | Could not read pid file, cluster "
                   "is probably not running.")
    while True:
        line = stopping.stderr.readline().strip().decode()
        print(line)
        if (stopped in line) or (not_running in line):
            break


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
