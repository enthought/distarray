# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import division

import unittest
import numpy
from random import randrange

from distarray.externals.six.moves import range

from distarray.dist.context import Context
from distarray.dist import maps


class TestClientMap(unittest.TestCase):

    def setUp(self):
        self.ctx = Context()

    def tearDown(self):
        self.ctx.close()

    def test_2D_bn(self):
        nrows, ncols = 31, 53
        cm = maps.Distribution.from_shape(self.ctx, (nrows, ncols),
                                          {0: 'b'}, (4, 1))
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertSequenceEqual(cm.owning_ranks((r,c)), [rank])

    def test_2D_bb(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = maps.Distribution.from_shape(self.ctx, (nrows, ncols), ('b', 'b'),
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
        cm = maps.Distribution.from_shape(self.ctx, (nrows, ncols), ('c', 'c'),
                                          (nprocs_per_dim, nprocs_per_dim))
        for r in range(nrows):
            for c in range(ncols):
                rank = (r % nprocs_per_dim) * nprocs_per_dim + (c % nprocs_per_dim)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])

    def test_is_compatible(self):
        nr, nc, nd = 10**5, 10**6, 10**4

        cm0 = maps.Distribution.from_shape(self.ctx, (nr, nc, nd),
                                           ('b', 'c', 'n'))
        self.assertTrue(cm0.is_compatible(cm0))

        cm1 = maps.Distribution.from_shape(self.ctx, (nr, nc, nd),
                                           ('b', 'c', 'n'))
        self.assertTrue(cm1.is_compatible(cm1))

        self.assertTrue(cm0.is_compatible(cm1))
        self.assertTrue(cm1.is_compatible(cm0))
        
        nr -= 1; nc -= 1; nd -= 1

        cm2 = maps.Distribution.from_shape(self.ctx, (nr, nc, nd),
                                           ('b', 'c', 'n'))

        self.assertFalse(cm1.is_compatible(cm2))
        self.assertFalse(cm2.is_compatible(cm1))

    def test_reduce(self):
        nr, nc, nd = 10**5, 10**6, 10**4

        dist = maps.Distribution.from_shape(
            self.ctx, (nr, nc, nd), ('b', 'c', 'n'), grid_shape=(2, 2, 1))

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
        dist = maps.Distribution.from_shape(self.ctx, (N,))
        new_dist = dist.reduce(axes=[0])
        self.assertEqual(new_dist.dist, ())
        self.assertSequenceEqual(new_dist.shape, ())
        self.assertEqual(new_dist.grid_shape, ())
        self.assertEqual(set(new_dist.targets), set(dist.targets[:1]))


class TestSlice(unittest.TestCase):

    def setUp(self):
        self.ctx = Context()

    def tearDown(self):
        self.ctx.close()

    def test_from_partial_slice_1d(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15,))

        s = (slice(0, 3),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, [0])
        self.assertSequenceEqual(d1.shape, (3,))

    def test_from_full_slice_1d(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15,))

        s = (slice(None),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertSequenceEqual(d1.maps[0].bounds, d0.maps[0].bounds)

    def test_from_full_slice_with_step_1d_0(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15,))

        s = (slice(None, None, 2),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertEqual(d1.maps[0].bounds[0][0], d0.maps[0].bounds[0][0])

    def test_from_full_slice_with_step_1d_1(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(30,))
        step = 4

        s = (slice(4, None, step),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertEqual(d1.maps[0].bounds[0][0], d0.maps[0].bounds[0][0])

    def test_from_full_slice_2d(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15, 20))

        s = (slice(None), slice(None))
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        for m0, m1 in zip(d0.maps, d1.maps):
            if m0.dist == 'b':
                self.assertSequenceEqual(m0.bounds, m1.bounds)
        self.assertSequenceEqual(d1.targets, d0.targets)

    def test_from_partial_slice_2d(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15, 20))

        s = (slice(3, 7), 4)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps)-1, len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist[:-1])
        for m, expected in zip(d1.maps, ([(0, 1), (1, 4)], [(0, 1)])):
            self.assertSequenceEqual(m.bounds, expected)

    def test_full_slice_with_int_2d(self):
        d0 = maps.Distribution.from_shape(context=self.ctx, shape=(15, 20))

        s = (slice(None), 4)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps)-1, len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist[:-1])
        self.assertEqual(d1.shape, (15,))
