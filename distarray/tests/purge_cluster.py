""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

from IPython.parallel import Client
from distarray.client import Context


def purge_cluster_namespaces():
    """ Remove all keys from the engine namespaces. """
    print('Purging namespaces on engines...')
    client = Client()
    view = client[:]
    context = Context(view)
    context.cleanup_keys_checked(False)


if __name__ == '__main__':
    purge_cluster_namespaces()

