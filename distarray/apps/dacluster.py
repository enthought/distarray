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

import argparse
import sys

from . import ipcluster_tools
from . import purge_cluster


def main():
    main_description = """
    Start, stop and manage a IPython.parallel cluster. `dacluster` can take
    all the commands IPython's `ipcluster` can, and a few extras that are
    distarray specific.
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

    purge_description = """
    Clear all the DistArray objects from the engines. This has a few
    leaks.
    """

    dump_description = """
    Print out key names that exist on the engines.
    """

    # subparses for all our commands
    parser_start = subparsers.add_parser('start',
                                         description=start_description)
    parser_stop = subparsers.add_parser('stop', description=stop_description)
    parser_restart = subparsers.add_parser('restart',
                                           description=restart_description)
    parser_clear = subparsers.add_parser('clear',
                                         description=clear_description)
    parser_purge = subparsers.add_parser('purge',
                                         description=purge_description)
    parser_dump = subparsers.add_parser('dump', description=dump_description)

    engine_help = """
    Number of engines to start.
    """

    # Add some optional arguments for `start` and `restart`
    parser_start.add_argument('-n', '--n', type=int, nargs='?', default=4,
                              help=engine_help)
    parser_restart.add_argument('-n', '--n', type=int, nargs='?', default=4,
                                help=engine_help)

    # set the functions each command should use
    parser_start.set_defaults(func=ipcluster_tools.start)
    parser_stop.set_defaults(func=ipcluster_tools.stop)
    parser_restart.set_defaults(func=ipcluster_tools.restart)
    parser_clear.set_defaults(func=ipcluster_tools.clear)
    parser_purge.set_defaults(func=purge_cluster.purge)
    parser_dump.set_defaults(func=purge_cluster.dump)

    # run it
    args = parser.parse_args()
    args.func(**vars(args))

if __name__ == '__main__':
    main()
