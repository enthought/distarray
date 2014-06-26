# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from __future__ import division

import unittest
from random import randrange

from distarray.externals.six.moves import range

from distarray.testing import ContextTestCase
from distarray.dist.maps import MapBase, Distribution


class TestClientMap(ContextTestCase):

    def test_2D_bn(self):
        nrows, ncols = 31, 53
        cm = Distribution.from_shape(self.context,
                                     (nrows, ncols),
                                     {0: 'b'},
                                     (4, 1))
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertSequenceEqual(cm.owning_ranks((r,c)), [rank])

    def test_2D_bb(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = Distribution.from_shape(self.context,
                                     (nrows, ncols),
                                     ('b', 'b'),
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
        cm = Distribution.from_shape(self.context,
                                     (nrows, ncols),
                                     ('c', 'c'),
                                     (nprocs_per_dim, nprocs_per_dim))
        for r in range(nrows):
            for c in range(ncols):
                rank = ((r % nprocs_per_dim) * nprocs_per_dim
                        + (c % nprocs_per_dim))
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])

    def test_is_compatible(self):
        nr, nc, nd = 10**5, 10**6, 10**4

        cm0 = Distribution.from_shape(self.context,
                                      (nr, nc, nd),
                                      ('b', 'c', 'n'))
        self.assertTrue(cm0.is_compatible(cm0))

        cm1 = Distribution.from_shape(self.context,
                                      (nr, nc, nd),
                                      ('b', 'c', 'n'))
        self.assertTrue(cm1.is_compatible(cm1))

        self.assertTrue(cm0.is_compatible(cm1))
        self.assertTrue(cm1.is_compatible(cm0))

        nr -= 1; nc -= 1; nd -= 1

        cm2 = Distribution.from_shape(self.context,
                                      (nr, nc, nd),
                                      ('b', 'c', 'n'))

        self.assertFalse(cm1.is_compatible(cm2))
        self.assertFalse(cm2.is_compatible(cm1))

    def test_is_compatible_nodist(self):
        # See GH issue #461.
        dist_bcn = Distribution.from_shape(self.context,
                                           (10, 10, 10),
                                           ('b', 'c', 'n'),
                                           (1,   1,   1),
                                           targets=[0])
        dist_nnn = Distribution.from_shape(self.context,
                                           (10, 10, 10),
                                           ('n', 'n', 'n'),
                                           (1,   1,   1),
                                           targets=[0])
        self.assertTrue(dist_bcn.is_compatible(dist_nnn))
        self.assertTrue(dist_nnn.is_compatible(dist_bcn))

    def test_is_compatible_degenerate(self):
        dist_bc = Distribution.from_shape(self.context,
                                          (10, 10),
                                          ('b', 'c'),
                                          (1,   1),
                                          targets=[0])
        dist_cb = Distribution.from_shape(self.context,
                                           (10, 10),
                                           ('c', 'b'),
                                           (1,   1),
                                           targets=[0])
        self.assertTrue(dist_bc.is_compatible(dist_cb))
        self.assertTrue(dist_cb.is_compatible(dist_bc))

    def test_is_compatible_degenerate_block_cyclic(self):
        size = 19937
        gdd_block_cyclic = (
                {
                    'dist_type': 'c',
                    'proc_grid_size': 1,
                    'block_size': 7,
                    'size': size,
                    },
                )
        gdd_block = (
                {
                    'dist_type': 'b',
                    'proc_grid_size': 1,
                    'bounds': [0, size],
                    },
                )
        gdd_cyclic = (
                {
                    'dist_type': 'c',
                    'proc_grid_size': 1,
                    'size': size,
                    },
                )
        dist_block_cyclic = Distribution(self.context, gdd_block_cyclic)
        dist_block = Distribution(self.context, gdd_block)
        dist_cyclic = Distribution(self.context, gdd_cyclic)

        self.assertTrue(dist_block_cyclic.is_compatible(dist_block))
        self.assertTrue(dist_block_cyclic.is_compatible(dist_cyclic))

        self.assertTrue(dist_block.is_compatible(dist_block_cyclic))
        self.assertTrue(dist_cyclic.is_compatible(dist_block_cyclic))

    def test_reduce(self):
        nr, nc, nd = 10**5, 10**6, 10**4

        dist = Distribution.from_shape(self.context,
                                       (nr, nc, nd),
                                       ('b', 'c', 'n'),
                                       grid_shape=(2, 2, 1))

        new_dist0 = dist.reduce(axes=[0])
        self.assertEqual(new_dist0.dist, ('c', 'n'))
        self.assertSequenceEqual(new_dist0.shape, (nc, nd))
        self.assertEqual(new_dist0.grid_shape, dist.grid_shape[1:])
        self.assertLess(set(new_dist0.targets), set(dist.targets))

        new_dist1 = dist.reduce(axes=[1])
        self.assertEqual(new_dist1.dist, ('b', 'n'))
        self.assertSequenceEqual(new_dist1.shape, (nr, nd))
        self.assertEqual(new_dist1.grid_shape,
                         dist.grid_shape[:1] + dist.grid_shape[2:])
        self.assertLess(set(new_dist1.targets), set(dist.targets))

        new_dist2 = dist.reduce(axes=[2])
        self.assertEqual(new_dist2.dist, ('b', 'c'))
        self.assertSequenceEqual(new_dist2.shape, (nr, nc))
        self.assertEqual(new_dist2.grid_shape, dist.grid_shape[:-1])
        self.assertEqual(set(new_dist2.targets), set(dist.targets))

    def test_reduce_0D(self):
        N = 10**5
        dist = Distribution.from_shape(self.context, (N,))
        new_dist = dist.reduce(axes=[0])
        self.assertEqual(new_dist.dist, ())
        self.assertSequenceEqual(new_dist.shape, ())
        self.assertEqual(new_dist.grid_shape, ())
        self.assertEqual(set(new_dist.targets), set(dist.targets[:1]))


class TestSlice(ContextTestCase):

    def test_from_partial_slice_1d(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15,))

        s = (slice(0, 3),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, [0])
        self.assertSequenceEqual(d1.shape, (3,))

    def test_from_full_slice_1d(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15,))

        s = (slice(None),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertSequenceEqual(d1.maps[0].bounds, d0.maps[0].bounds)

    def test_from_full_slice_with_step_1d_0(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15,))

        s = (slice(None, None, 2),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertEqual(d1.maps[0].bounds[0][0], d0.maps[0].bounds[0][0])

    def test_from_full_slice_with_step_1d_1(self):
        d0 = Distribution.from_shape(context=self.context, shape=(30,))
        step = 4

        s = (slice(4, None, step),)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        self.assertSequenceEqual(d1.targets, d0.targets)
        self.assertEqual(d1.maps[0].bounds[0][0], d0.maps[0].bounds[0][0])

    def test_from_full_slice_2d(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15, 20))

        s = (slice(None), slice(None))
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps), len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist)
        for m0, m1 in zip(d0.maps, d1.maps):
            if m0.dist == 'b':
                self.assertSequenceEqual(m0.bounds, m1.bounds)
        self.assertSequenceEqual(d1.targets, d0.targets)

    def test_from_partial_slice_2d(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15, 20))

        s = (slice(3, 7), 4)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps)-1, len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist[:-1])
        for m, expected in zip(d1.maps, ([(0, 1), (1, 4)], [(0, 1)])):
            self.assertSequenceEqual(m.bounds, expected)

    def test_full_slice_with_int_2d(self):
        d0 = Distribution.from_shape(context=self.context, shape=(15, 20))

        s = (slice(None), 4)
        d1 = d0.slice(s)

        self.assertEqual(len(d0.maps)-1, len(d1.maps))
        self.assertSequenceEqual(d1.dist, d0.dist[:-1])
        self.assertEqual(d1.shape, (15,))


class TestDunderMethods(ContextTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDunderMethods, cls).setUpClass()
        cls.shape = (3, 4, 5, 6)
        cls.cm = Distribution.from_shape(cls.context, cls.shape)

    def test___len__(self):
        self.assertEqual(len(self.cm), 4)

    def test___getitem__(self):
        for m in self.cm:
            self.assertTrue(isinstance(m, MapBase))

        self.assertEqual(self.cm[0].dist, 'b')
        self.assertEqual(self.cm[1].dist, 'n')
        self.assertEqual(self.cm[2].dist, 'n')
        self.assertEqual(self.cm[-1].dist, 'n')


class TestDistributionCreation(ContextTestCase):
    def test_all_n_dist(self):
        distribution = Distribution.from_shape(self.context,
                                               shape=(3, 3),
                                               dist=('n', 'n'))
        self.context.ones(distribution)


if __name__ == '__main__':
    unittest.main(verbosity=2)
