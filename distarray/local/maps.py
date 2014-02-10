# encoding: utf-8

from __future__ import division

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

from six.moves import range, zip
from math import ceil


def no_distribution(dd):
    """Return the global indicies owned by this undistributed process.

    Requires the 'size' key.
    """
    return range(dd['size'])


def block(dd):
    """Return the global indices owned by this block-distributed process.

    Requires 'start' and 'stop' keys.
    """
    return range(dd['start'], dd['stop'])


def cyclic(dd):
    """Return the global indices owned by this (block-)cyclically-distributed
    process.

    Requires 'start', 'size', 'proc_grid_size', and (optionally) 'block_size'
    keys.  If 'block_size' key does not exist, it is set to 1.
    """
    dd.setdefault('block_size', 1)
    nblocks = int(ceil(dd['size'] / dd['block_size']))
    block_indices = range(0, nblocks, dd['proc_grid_size'])

    global_indices = []
    for block_index in block_indices:
        block_start = block_index * dd['block_size'] + dd['start']
        block_stop = block_start + dd['block_size']
        block = range(block_start, min(block_stop, dd['size']))
        global_indices.extend(block)

    return global_indices


def unstructured(dd):
    """Return the arbitrary global indices owned by this  process.

    Requires the 'indices' key.
    """
    return dd['indices']


dist_type_to_global_indices = {
    'n': no_distribution,
    'b': block,
    'c': cyclic,
    'u': unstructured,
}


class IndexMap(object):

    """Provide global->local and local->global index mappings.

    Attributes
    ----------
    global_index : list of int or range object
        Given a local index as a key, return the corresponding global index.
    local_index : dict of int -> int
        Given a global index as a key, return the corresponding local index.
    """

    def __init__(self, global_indices):
        """Make an IndexMap from a local_index and global_index.

        Parameters
        ----------
        global_indices: list of int or range object
            Each position contains the corresponding global index for a
            local index (position).
        """
        self.global_index = global_indices
        local_indices = range(len(global_indices))
        self.local_index = dict(zip(global_indices, local_indices))

    @property
    def size(self):
        return len(self.global_index)

    @classmethod
    def from_dimdict(cls, dimdict):
        """Make an IndexMap from a `dimdict` data structure."""
        global_indices_fn = dist_type_to_global_indices[dimdict['dist_type']]
        return IndexMap(global_indices_fn(dimdict))
