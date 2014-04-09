#!/usr/bin/env python

import argparse
import sys

import ipcluster_tools
import purge_cluster


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # We failed parsing the args, pass them directly to ipcluster
        # to see if it can handle them.
        ipcluster_tools.run_ipcluster(sys.argv[1:])


description = """
Start, stop and manage a IPython.parallel cluster. `dacluster` can take
all the commands IPython's `ipcluster` can, and a few extras that are
distarray specific.
"""
parser = ArgumentParser(description=description)

# Print help if no command line args are supplied
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)


subparsers = parser.add_subparsers(help='subparsers')

# subparses for all our commands
parser_start = subparsers.add_parser('start')
parser_stop = subparsers.add_parser('stop')
parser_restart = subparsers.add_parser('restart')
parser_clear = subparsers.add_parser('clear')
parser_purge = subparsers.add_parser('purge')

# set the functions each command should use
parser_start.set_defaults(func=ipcluster_tools.start)
parser_stop.set_defaults(func=ipcluster_tools.stop)
parser_restart.set_defaults(func=ipcluster_tools.restart)
parser_clear.set_defaults(func=ipcluster_tools.clear)
parser_purge.set_defaults(func=purge_cluster.purge)


def main():
    args = parser.parse_args()
    args.func()
