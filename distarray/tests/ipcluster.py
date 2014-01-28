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

    You have to wait for it to start, however.
    """
    # FIXME: This should be reimplemented to signal when the cluster has
    # successfully started

    engines = "--engines=MPIEngineSetLauncher"
    Popen([ipcluster_cmd, 'start', '-n', str(n), engines, str('&')],
           stdout=PIPE, stderr=PIPE)


def stop():
    """Convenient way to stop an ipcluster."""

    Popen([ipcluster_cmd, 'stop'], stdout=PIPE, stderr=PIPE)


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
    fn()
