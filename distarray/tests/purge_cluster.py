""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

import sys

from IPython.parallel import Client
from distarray.context import Context


def dump():
    """ Print out all names that exist on the engines. """
    client = Client()
    view = client[:]
    context = Context(view)
    context.dump_keys()


def purge():
    """ Remove all keys from the engine namespaces. """
    print('Purging namespaces on engines...')
    client = Client()
    view = client[:]
    context = Context(view)
    context.purge_keys()


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
