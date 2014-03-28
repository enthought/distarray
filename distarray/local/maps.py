# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from __future__ import division

from distarray.externals.six.moves import range, zip
from math import ceil


def not_distributed(dd):
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


global_indices_from_dist_type = {
    'n': not_distributed,
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

    # def __init__(self, global_indices):
        # pass

    def _internal__init__(self, global_indices):
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
        global_indices_fn = global_indices_from_dist_type[dimdict['dist_type']]
        self = cls.__new__(cls)
        self._internal__init__(global_indices_fn(dimdict))
        return self


class BlockMap(IndexMap):

    dist = 'b'

    def __init__(self, global_size, grid_size, grid_rank, start, stop):
        self.start = start
        self.stop = stop
        self.local_size = stop - start
        self.global_size = global_size
        self.grid_size = grid_size
        self.grid_rank = grid_rank

    def local_from_global(self, gidx):
        if gidx < self.start or gidx >= self.stop:
            raise IndexError()
        return gidx - self.start

    def global_from_local(self, lidx):
        if lidx >= self.local_size:
            raise IndexError()
        return lidx + self.start

    def global_slice(self):
        return slice(self.start, self.stop)

    @property
    def dim_dict(self):
        return {'dist_type': self.dist,
                'size': self.global_size,
                'proc_grid_rank': self.grid_rank,
                'proc_grid_size': self.grid_size,
                'start': self.start,
                'stop': self.stop,
                }


class CyclicMap(IndexMap):

    dist = 'c'

    def __init__(self, global_size, grid_size, grid_rank, start):
        if start != grid_rank:
            raise ValueError()
        if start >= grid_size:
            msg = "start (%d) is greater or equal to grid_size (%d)"
            raise ValueError(msg % (start, grid_size))
        self.start = start
        self.grid_size = grid_size
        self.grid_rank = grid_rank

        self.local_size = (global_size - 1 - grid_rank) // grid_size + 1
        self.global_size = global_size

    def local_from_global(self, gidx):
        if (gidx - self.start) % self.grid_size:
            raise IndexError()
        return (gidx - self.start) // self.grid_size

    def global_from_local(self, lidx):
        if lidx >= self.local_size:
            raise IndexError()
        return (lidx * self.grid_size) + self.start

    def global_slice(self):
        return slice(self.start, self.global_size, self.grid_size)

    @property
    def dim_dict(self):
        return {'dist_type': self.dist,
                'size': self.global_size,
                'proc_grid_rank': self.grid_rank,
                'proc_grid_size': self.grid_size,
                'start': self.start,
                }


class BlockCyclicMap(IndexMap):

    dist = 'c'
    
    def __init__(self, global_size, grid_size, grid_rank, start, block_size):
        if start % block_size:
            raise ValueError()
        self.start_block = start // block_size
        self.block_size = block_size
        global_nblocks = global_size // block_size
        if global_nblocks * block_size != global_size:
            raise ValueError()
        self.grid_size = grid_size

        local_nblocks = (global_nblocks - 1 - grid_rank) // grid_rank + 1
        self.local_size = local_nblocks * block_size
        self.global_size = global_size

        # if global_nblocks == grid_size * (global_nblocks // grid_size):
            # local_nblocks = global_nblocks // grid_size
        # elif grid_rank == grid_size - 1:
            # local_nblocks = global_nblocks % grid_size
        # else:
            # local_nblocks = global_nblocks // grid_size + 1
        # self.local_size = local_nblocks * block_size

    def local_from_global(self, gidx):
        global_block, offset = divmod(gidx, self.block_size)
        if (global_block - self.start_block) % self.grid_size:
            raise IndexError()
        return self.block_size * ((global_block - self.start_block) // self.grid_size) + offset

    def global_from_local(self, lidx):
        if lidx >= self.local_size:
            raise IndexError()
        local_block, offset = divmod(lidx, self.block_size)
        global_block = (local_block * self.grid_size) + self.start_block
        return global_block * self.block_size + offset

    def global_slice(self):
        raise NotImplementedError()
        # TODO: FIXME: Not really a slice, but a fancy index, since can't
        # represent blockcyclic as a slice!
        # idxs = np.empty(self.size, dtype=np.int)
        # for offset in range(self.block_size):

    @property
    def dim_dict(self):
        return {'dist_type': self.dist,
                'size': self.global_size,
                'proc_grid_rank': self.grid_rank,
                'proc_grid_size': self.grid_size,
                'start': self.start,
                'block_size': self.block_size,
                }



class UnstructuredMap(IndexMap):

    dist = 'u'

    def __init__(self, global_size, grid_size, grid_rank, indices):
        self.global_size = global_size
        self.grid_size = grid_size
        self.grid_rank = grid_rank
        self.indices = list(indices)
        self.local_size = len(self.indices)

    def local_from_global(self, gidx):
        try:
            lidx = self.indices.index(gidx)
        except ValueError:
            raise IndexError()
        return lidx

    def global_from_local(self, lidx):
        return self.indices[lidx]

    def global_slice(self):
        raise NotImplementedError()

    @property
    def dim_dict(self):
        return {'dist_type': self.dist,
                'size': self.global_size,
                'proc_grid_rank': self.grid_rank,
                'proc_grid_size': self.grid_size,
                'indices': self.indices,
                }
