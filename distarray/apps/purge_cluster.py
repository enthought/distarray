# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

import sys

from distarray.context import Context


def dump(*args, **kwargs):
    """ Print out key names that exist on the engines. """
    context = Context()
    keylist = context.dump_keys(all_other_contexts=True)
    num_keys = len(keylist)
    print('*** %d ENGINE KEYS ***' % (num_keys))
    for key, targets in keylist:
        print('%s : %r' % (key, targets))


def purge(*args, **kwargs):
    """ Remove keys from the engine namespaces. """
    print('Purging keys from engines...')
    context = Context()
    context.cleanup(all_other_contexts=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dump':
        dump()
    elif cmd == 'purge':
        purge()
    else:
        raise ValueError("%s command not found" % (cmd,))
