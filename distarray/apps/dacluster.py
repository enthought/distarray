#!/usr/bin/env python
# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Start, stop and manage a IPython.parallel cluster. `dacluster` can take
all the commands IPython's `ipcluster` can, and a few extras that are
distarray specific.
"""

from __future__ import print_function

import argparse
import sys
from time import sleep
from subprocess import Popen, PIPE

from distarray.externals import six
from distarray.globalapi.ipython_cleanup import clear_all


is_anaconda = "Anaconda" in sys.version or "Continuum" in sys.version

if six.PY2 or is_anaconda:
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


def clear(**kwargs):
    """ Removes all distarray-related modules from engines' sys.modules."""
    mods = clear_all()

    msg = "*** Removing %d distarray modules from engines' namespace. ***"
    print(msg % len(list(mods.values())[0]))


def main():
    """ Main function for dacluster utility.

    Either start, stop, restart, or clear is called depending on the
    command line arguments.
    """
    main_description = """
    Start, stop and manage a IPython.parallel cluster. `dacluster` can take
    all the commands IPython's `ipcluster` can, and a few extras that are
    distarray specific. For details on a subcommand, try `dacluster
    <subcommand> --help`.
    """
    parser = argparse.ArgumentParser(description=main_description)

    # Print help if no command line args are supplied
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    subparsers = parser.add_subparsers()

    start_description = """
    Start a new IPython.parallel cluster.
    """

    stop_description = """
    Stop a IPython.parallel cluster.
    """

    restart_description = """
    Restart a IPython.parallel cluster.
    """

    clear_description = """
    Clear the namespace and imports on the cluster. This should be the
    same as restarting the engines, but faster.
    """
    # subparses for all our commands
    parser_start = subparsers.add_parser('start',
                                         description=start_description)
    parser_stop = subparsers.add_parser('stop', description=stop_description)
    parser_restart = subparsers.add_parser('restart',
                                           description=restart_description)
    parser_clear = subparsers.add_parser('clear',
                                         description=clear_description)

    engine_help = """
    Number of engines to start.
    """

    # Add some optional arguments for `start` and `restart`
    parser_start.add_argument('-n', '--n', type=int, nargs='?', default=4,
                              help=engine_help)
    parser_restart.add_argument('-n', '--n', type=int, nargs='?', default=4,
                                help=engine_help)

    # set the functions each command should use
    parser_start.set_defaults(func=start)
    parser_stop.set_defaults(func=stop)
    parser_restart.set_defaults(func=restart)
    parser_clear.set_defaults(func=clear)

    # run it
    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == '__main__':
    main()
