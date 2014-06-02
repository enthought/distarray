# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from random import randrange

from distarray.externals.six.moves import range

from distarray.dist.context import Context
from distarray.dist import maps as client_map


class TestClientMap(unittest.TestCase):

    def setUp(self):
        self.ctx = Context()

    def tearDown(self):
        self.ctx.close()

    def test_2D_bn(self):
        if len(self.ctx.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
        nrows, ncols = 31, 53
        cm = client_map.Distribution.from_shape(self.ctx,
                (nrows, ncols),
                {0: 'b'},
                (4, 1),
                self.ctx.targets[:4])
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertSequenceEqual(cm.owning_ranks((r,c)), [rank])

    def test_2D_bb(self):
        if len(self.ctx.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.Distribution.from_shape(
                self.ctx, (nrows, ncols), ('b', 'b'),
                (nprocs_per_dim, nprocs_per_dim),
                targets=self.ctx.targets[:4])
        row_chunks = nrows // nprocs_per_dim + 1
        col_chunks = ncols // nprocs_per_dim + 1
        for r in range(nrows):
            for c in range(ncols):
                rank = (r // row_chunks) * nprocs_per_dim + (c // col_chunks)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])

    def test_2D_cc(self):
        if len(self.ctx.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.Distribution.from_shape(
                self.ctx, (nrows, ncols), ('c', 'c'),
                (nprocs_per_dim, nprocs_per_dim),
                self.ctx.targets[:4])
        for r in range(nrows):
            for c in range(ncols):
                rank = (r % nprocs_per_dim) * nprocs_per_dim + (c % nprocs_per_dim)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])


    def test_is_compatible(self):
        nr, nc, nd = 10**5, 10**6, 10**4

        cm0 = client_map.Distribution.from_shape(
                 self.ctx, (nr, nc, nd), ('b', 'c', 'n'))
        self.assertTrue(cm0.is_compatible(cm0))

        cm1 = client_map.Distribution.from_shape(
                 self.ctx, (nr, nc, nd), ('b', 'c', 'n'))
        self.assertTrue(cm1.is_compatible(cm1))

        self.assertTrue(cm0.is_compatible(cm1))
        self.assertTrue(cm1.is_compatible(cm0))
        
        nr -= 1; nc -= 1; nd -= 1

        cm2 = client_map.Distribution.from_shape(
                 self.ctx, (nr, nc, nd), ('b', 'c', 'n'))

        self.assertFalse(cm1.is_compatible(cm2))
        self.assertFalse(cm2.is_compatible(cm1))

    def test_reduce(self):
        if len(self.ctx.targets) < 4:
            raise unittest.SkipTest("not enough targets to run test.")

        nr, nc, nd = 10**5, 10**6, 10**4

        dist = client_map.Distribution.from_shape(
                 self.ctx, (nr, nc, nd), ('b', 'c', 'n'), 
                 grid_shape=(2, 2, 1),
                 targets=self.ctx.targets[:4])

        new_dist0 = dist.reduce(axes=[0])
        self.assertEqual(new_dist0.dist, ('c', 'n'))
        self.assertSequenceEqual(new_dist0.shape, (nc, nd))
        self.assertEqual(new_dist0.grid_shape, dist.grid_shape[1:])
        self.assertLess(set(new_dist0.targets), set(dist.targets))

        new_dist1 = dist.reduce(axes=[1])
        self.assertEqual(new_dist1.dist, ('b', 'n'))
        self.assertSequenceEqual(new_dist1.shape, (nr, nd))
        self.assertEqual(new_dist1.grid_shape, dist.grid_shape[:1]+dist.grid_shape[2:])
        self.assertLess(set(new_dist1.targets), set(dist.targets))

        new_dist2 = dist.reduce(axes=[2])
        self.assertEqual(new_dist2.dist, ('b', 'c'))
        self.assertSequenceEqual(new_dist2.shape, (nr, nc))
        self.assertEqual(new_dist2.grid_shape, dist.grid_shape[:-1])
        self.assertEqual(set(new_dist2.targets), set(dist.targets))

    def test_reduce_0D(self):
        N = 10**5
        dist = client_map.Distribution.from_shape(self.ctx, (N,))
        new_dist = dist.reduce(axes=[0])
        self.assertEqual(new_dist.dist, ())
        self.assertSequenceEqual(new_dist.shape, ())
        self.assertEqual(new_dist.grid_shape, ())
        self.assertEqual(set(new_dist.targets), set(dist.targets[:1]))
