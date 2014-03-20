# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

import sys

from IPython.parallel import Client
from distarray.context import Context


def dump():
    """ Print out key names that exist on the engines. """
    client = Client()
    context = Context(client)
    keylist = context.dump_keys(all_other_contexts=True)
    num_keys = len(keylist)
    print('*** %d ENGINE KEYS ***' % (num_keys))
    for key, targets in keylist:
        print('%s : %r' % (key, targets))


def purge():
    """ Remove keys from the engine namespaces. """
    print('Purging keys from engines...')
    client = Client()
    context = Context(client)
    context.purge_keys(all_other_contexts=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dump':
        dump()
    elif cmd == 'purge':
        purge()
    else:
        raise ValueError("%s command not found" % (cmd,))
