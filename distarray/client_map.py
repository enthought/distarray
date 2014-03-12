# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

from itertools import product
from distarray.local.base import _start_stop_block

import numpy as np

class ClientMap(object):
    '''
    Governs the mapping between global indices and process ranks.

    Works with the LocalMap classes to facilitate communication between global
    and local processes.

    Invariants: TODO

    '''

    # Does proc-grid data need to be in here?

    # How does the DAP metadata need to be represented on client-side?

    # Need to hold on to mapping between proc_grid_rank and global indices.

    # Need to hold on to mapping between process rank and proc_grid tuple.

    def __init__(self, shape, dist, grid_shape):
        self.shape = shape
        self.dist = dist
        self.grid_shape = tuple(grid_shape) + (1,) * (len(shape) - len(grid_shape))

        nelts = reduce(int.__mul__, grid_shape)
        self.rank_from_coords = np.arange(nelts).reshape(*self.grid_shape)

        # TODO: FIXME: assert that self.rank_from_coords is valid and conforms
        # to how MPI does it.

        # only handle the easy cases for now...
        for dtype in dist.values():
            assert dtype in ('b', 'n')

        self._set_dim_bounds()

    def _set_dim_bounds(self):
        # FIXME: only works for 'b' and 'n'
        self.dim_bounds = []
        for dim, (size, grid_size) in enumerate(zip(self.shape, self.grid_shape)):
            disttype = self.dist.get(dim, 'n')
            if disttype == 'n':
                self.dim_bounds.append([(0, size)])
            elif disttype == 'b':
                bounds = [_start_stop_block(size, grid_size, grid_rank)
                        for grid_rank in range(grid_size)]
                self.dim_bounds.append(bounds)
            else:
                raise NotImplementedError('distype %r not yet supported' % disttype)

    def possibly_owning_ranks(self, idxs):
        # build up coords
        coord_zip = []
        for dim, idx in enumerate(idxs):
            coords = []
            bounds = self.dim_bounds[dim]
            assert isinstance(bounds, list), "bounds not list"
            assert isinstance(bounds[0], tuple), "bounds not tuple"
            for (coord, (lower, upper)) in enumerate(bounds):
                if idx >= lower and idx < upper:
                    coords.append(coord)
            assert coords, "coords not empty"
            coord_zip.append(coords)
        all_coords = product(*coord_zip)
        ranks = [self.rank_from_coords[c] for c in all_coords]
        return ranks
