# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from distarray import Context
from distarray.local import maps
from distarray import client_map
from random import randrange

from distarray.externals.six.moves import range


class TestClientMap(unittest.TestCase):

    def setUp(self):
        self.ctx = Context()

    def tearDown(self):
        self.ctx.cleanup()

    def test_2D_bn(self):
        nrows, ncols = 31, 53
        cm = client_map.ClientMDMap(self.ctx, (nrows, ncols), {0:'b'}, (4,1))
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertSequenceEqual(cm.owning_ranks((r,c)), [rank])

    def test_2D_bb(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.ClientMDMap(self.ctx, (nrows, ncols), ('b', 'b'),
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
        cm = client_map.ClientMDMap(self.ctx, (nrows, ncols), ('c', 'c'),
                (nprocs_per_dim, nprocs_per_dim))
        for r in range(nrows):
            for c in range(ncols):
                rank = (r % nprocs_per_dim) * nprocs_per_dim + (c % nprocs_per_dim)
                actual = cm.owning_ranks((r,c))
                self.assertSequenceEqual(actual, [rank])



class TestNotDistMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='n', size=20)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global(self):
        gis = range(0, 20)
        lis = [self.m.local_from_global(gi) for gi in gis]
        expected = list(range(20))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_IndexError(self):
        gi = 20
        self.assertRaises(IndexError, self.m.local_from_global, gi)

    def test_global_from_local(self):
        lis = range(20)
        gis = [self.m.global_from_local(li) for li in lis]
        expected = list(range(20))
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_IndexError(self):
        li = 20
        self.assertRaises(IndexError, self.m.global_from_local, li)


class TestBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='b', size=(39-16), start=16, stop=39)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global(self):
        gis = range(16, 39)
        lis = [self.m.local_from_global(gi) for gi in gis]
        expected = list(range(23))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_IndexError(self):
        gi = 15
        self.assertRaises(IndexError, self.m.local_from_global, gi)

        gi = 39
        self.assertRaises(IndexError, self.m.local_from_global, gi)

    def test_global_from_local(self):
        lis = range(23)
        gis = [self.m.global_from_local(li) for li in lis]
        expected = list(range(16, 39))
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_IndexError(self):
        li = 25
        self.assertRaises(IndexError, self.m.global_from_local, li)


class TestCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4, proc_grid_rank=2)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global(self):
        gis = (2, 6, 10, 14)
        lis = [self.m.local_from_global(gi) for gi in gis]
        expected = tuple(range(4))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_IndexError(self):
        gi = 3
        self.assertRaises(IndexError, self.m.local_from_global, gi)

        gi = 7
        self.assertRaises(IndexError, self.m.local_from_global, gi)

    def test_global_from_local(self):
        lis = range(4)
        gis = [self.m.global_from_local(li) for li in lis]
        expected = (2, 6, 10, 14)
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_from_local, li)


class TestBlockCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4,
                       block_size=2)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global(self):
        """Test the local_index method of BlockCyclicMap."""
        gis = (2, 3, 10, 11)
        lis = [self.m.local_from_global(gi) for gi in gis]
        expected = tuple(range(4))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_IndexError(self):
        gi = 4
        self.assertRaises(IndexError, self.m.local_from_global, gi)
        gi = 12
        self.assertRaises(IndexError, self.m.local_from_global, gi)

    def test_global_from_local(self):
        lis = range(4)
        gis = [self.m.global_from_local(li) for li in lis]
        expected = (2, 3, 10, 11)
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_from_local, li)


class TestMapEquivalences(unittest.TestCase):

    def test_compare_bcm_bm_local_index(self):
        """Test Block-Cyclic against Block map."""
        start = 4
        size = 16
        grid = 4
        block = size // grid
        dimdict = dict(start=start, size=size, proc_grid_size=grid)

        bcm = maps.map_from_dim_dict(dict(list(dimdict.items()) +
                                              [('dist_type', 'c'),
                                               ('block_size', block)]))
        bm = maps.map_from_dim_dict(dict(list(dimdict.items()) +
                                             [('dist_type', 'b'),
                                              ('stop', size // grid +
                                                       start)]))
        bcm_lis = [bcm.local_from_global(e) for e in range(4, 8)]
        bm_lis = [bm.local_from_global(e) for e in range(4, 8)]
        self.assertSequenceEqual(bcm_lis, bm_lis)

    def test_compare_bcm_cm_local_index(self):
        """Test Block-Cyclic against Cyclic map."""
        start = 1
        size = 16
        grid = 4
        block = 1
        dimdict = dict(start=start, size=size, proc_grid_size=grid,
                       block_size=block, proc_grid_rank=start)
        bcm = maps.map_from_dim_dict(dict(list(dimdict.items()) +
                                              [('dist_type', 'c')]))
        cm = maps.map_from_dim_dict(dict(list(dimdict.items()) +
                                             [('dist_type', 'c')]))
        bcm_lis = [bcm.local_from_global(e) for e in range(1, 16, 4)]
        cm_lis = [cm.local_from_global(e) for e in range(1, 16, 4)]
        self.assertSequenceEqual(bcm_lis, cm_lis)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
