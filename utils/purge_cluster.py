# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

""" Simple utility to clean out existing namespaces on engines. """

from __future__ import print_function

import sys

from distarray.context import DISTARRAY_BASE_NAME
from distarray.cleanup import cleanup, get_local_keys
from distarray.ipython_utils import IPythonClient


def dump(view, prefix):
    """ Print out key names that exist on the engines. """
    targets_from_key = get_local_keys(view, prefix)
    num_keys = len(targets_from_key)
    print('*** %d ENGINE KEYS ***' % (num_keys))
    for key, targets in sorted(targets_from_key):
        print('%s : %r' % (key, targets))

def purge(view, prefix):
    """ Remove keys from the engine namespaces. """
    print('Purging keys from engines...')
    cleanup(view=view, prefix=prefix)


if __name__ == '__main__':
    cmd = sys.argv[1]
    client = IPythonClient()
    view = client[:]
    prefix = DISTARRAY_BASE_NAME
    if cmd == 'dump':
        dump(view, prefix)
    elif cmd == 'purge':
        purge(view, prefix)
    else:
        raise ValueError("%s command not found" % (cmd,))
