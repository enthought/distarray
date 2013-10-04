import six
import time
from subprocess import Popen, PIPE


def run_ipcluster(n=4):
    if six.PY2:
        cmd = 'ipcluster'
    elif six.PY3:
        cmd = 'ipcluster3'
    else:
        raise NotImplementedError("Not run with Python 2 *or* 3?")

    engines = "--engines=MPIEngineSetLauncher"
    rval = Popen([cmd, 'start', '-n', str(n), engines, str('&')],
                 stdout=PIPE, stderr=PIPE)
    time.sleep(30)  # FIXME: this is a hack; how do we know when engines have
                    # been started successfully?


if __name__ == '__main__':
    run_ipcluster()
