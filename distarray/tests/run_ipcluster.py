import six
from subprocess import Popen, PIPE


def run_ipcluster(n=4):
    """Convenient way to start an ipcluster for testing.

    You have to wait for it to start, however.
    """
    # FIXME: This should be reimplemented to signal when the cluster has
    # successfully started
    if six.PY2:
        cmd = 'ipcluster'
    elif six.PY3:
        cmd = 'ipcluster3'
    else:
        raise NotImplementedError("Not run with Python 2 *or* 3?")

    engines = "--engines=MPIEngineSetLauncher"
    rval = Popen([cmd, 'start', '-n', str(n), engines, str('&')],
                 stdout=PIPE, stderr=PIPE)


if __name__ == '__main__':
    run_ipcluster()
