# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from random import randrange

from distarray.externals.six.moves import range

from distarray.client.context import Context
from distarray.client import maps as client_map


class TestClientMap(unittest.TestCase):

    def setUp(self):
        self.ctx = Context()

    def tearDown(self):
        self.ctx.close()

    def test_2D_bn(self):
        nrows, ncols = 31, 53
        cm = client_map.Distribution.from_shape(self.ctx, (nrows, ncols),
                                                {0: 'b'}, (4, 1))
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertSequenceEqual(cm.owning_ranks((r,c)), [rank])

    def test_2D_bb(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.Distribution.from_shape(
                self.ctx, (nrows, ncols), ('b', 'b'),
                (nprocs_per_dim, nprocs_per_dim))
        row_chunks = nrows // nprocs_per_dim + 1
        col_chunks = ncols // nprocs_per_dim + 1
        for r in range(nrows):
            for c in range(ncols):
                rank = (r // row_chunks) * nprocs_per_dim + (c // col_chunks)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])

    def test_2D_cc(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.Distribution.from_shape(
                self.ctx, (nrows, ncols), ('c', 'c'),
                (nprocs_per_dim, nprocs_per_dim))
        for r in range(nrows):
            for c in range(ncols):
                rank = (r % nprocs_per_dim) * nprocs_per_dim + (c % nprocs_per_dim)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])



