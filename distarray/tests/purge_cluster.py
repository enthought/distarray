""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

import sys

from IPython.parallel import Client
from distarray.context import Context


def dump():
    """ Print out all names that exist on the engines. """
    client = Client()
    context = Context(client)
    keylist = context.dump_keys(all_contexts=True)
    print('*** ENGINE KEYS ***')
    for key, targets in keylist:
        print('%s : %r' % (key, targets))


def purge():
    """ Remove all keys from the engine namespaces. """
    print('Purging keys from engines...')
    client = Client()
    context = Context(client)
    print('Purge context:')
    print('    key context:', context.key_context)
    print(''     'comm key:', context._comm_key)
    context.purge_keys(all_contexts=True)
    print('Did purge_keys...')


if __name__ == '__main__':
    cmd = sys.argv[1]
    fn = eval(cmd)
