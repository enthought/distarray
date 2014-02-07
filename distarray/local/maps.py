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


def block(dd):
    """Return the global indices owned by this block-distributed process.

    For a regularly-distributed block distribution, 'gridrank', 'size',
    and 'proc_grid_size' keys are required.  For an irregularly-distributed block
    distribution, 'start' and 'stop' are required.
    """
    return range(dd['start'], dd['stop'])


def cyclic(dd):
    """Return the global indices owned by this cyclically-distributed process.

    Requires 'start', 'size', and 'proc_grid_size' keys.
    """
    return range(dd['start'], dd['size'], dd['proc_grid_size'])


def block_cyclic(dd):
    """Return the global indices owned by this block-cyclically-distributed
    process.

    Requires 'start', 'size', 'proc_grid_size', and 'blocksize' keys.
    """
    nblocks = int(ceil(dd['size'] / dd['blocksize']))
    block_indices = range(0, nblocks, dd['proc_grid_size'])

    global_indices = []
    for block_index in block_indices:
        block_start = block_index * dd['blocksize'] + dd['start']
        block_stop = block_start + dd['blocksize']
        block = range(block_start, min(block_stop, dd['size']))
        global_indices.extend(block)

    return global_indices


def unstructured(dd):
    """Return the arbitrary global indices owned by this  process.

    Requires the 'indices' key.
    """
    return dd['indices']


disttype_to_global_indices = {
    'b': block,
    'bp': block,
    'c': cyclic,
    'bc': block_cyclic,
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
        global_indices_fn = disttype_to_global_indices[dimdict['disttype']]
        return IndexMap(global_indices_fn(dimdict))
