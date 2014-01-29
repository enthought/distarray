"""
Simple runner for `ipcluster start` or `ipcluster stop` on Python 2 or 3, as
appropriate.
"""

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

    match = six.text_type("Engines appear to have started successfully")
    while True:
        line = six.text_type(cluster.stderr.readline())
        if match in line:
            break


def stop():
    """Convenient way to stop an ipcluster."""
    Popen([ipcluster_cmd, 'stop'], stdout=PIPE, stderr=PIPE)


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
    fn()
