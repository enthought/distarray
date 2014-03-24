# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import unittest
from distarray.testing import IpclusterTestCase
from distarray import Context
from distarray.local import maps
from distarray import client_map
from random import randrange

from distarray.externals.six.moves import range

class TestClientMap(IpclusterTestCase):

    def setUp(self):
        self.ctx = Context(self.client)

     # overloads base class...
    def tearDown(self):
        del self.ctx
        super(TestClientMap, self).tearDown()

    def test_2D_bn(self):
        nrows, ncols = 31, 53
        cm = client_map.ClientMDMap(self.ctx, (nrows, ncols), {0:'b'}, (4,1))
        chunksize = (nrows // 4) + 1
        for _ in range(100):
            r, c = randrange(nrows), randrange(ncols)
            rank = r // chunksize
            self.assertEqual(cm.owning_ranks((r,c)), [rank])

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
                self.assertEqual(actual, [rank])

    def test_2D_cc(self):
        nrows, ncols = 3, 5
        nprocs_per_dim = 2
        cm = client_map.ClientMDMap(self.ctx, (nrows, ncols), ('c', 'c'),
                (nprocs_per_dim, nprocs_per_dim))
        for r in range(nrows):
            for c in range(ncols):
                rank = (r % nprocs_per_dim) * nprocs_per_dim + (c % nprocs_per_dim)
                actual = cm.owning_ranks((r,c))
                self.assertEqual(actual, [rank])



class TestNotDistMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='n', size=20)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_local_index(self):
        gis = range(0, 20)
        lis = list(self.m.local_index[gi] for gi in gis)
        expected = list(range(20))
        self.assertEqual(lis, expected)

    def test_local_index_KeyError(self):
        gi = 20
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

    def test_global_index(self):
        lis = range(20)
        gis = list(self.m.global_index[li] for li in lis)
        expected = list(range(20))
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = 20
        self.assertRaises(IndexError, self.m.global_index.__getitem__, li)


class TestBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='b', start=16, stop=39)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_local_index(self):
        gis = range(16, 39)
        lis = list(self.m.local_index[gi] for gi in gis)
        expected = list(range(23))
        self.assertEqual(lis, expected)

    def test_local_index_KeyError(self):
        gi = 15
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

        gi = 39
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

    def test_global_index(self):
        lis = range(23)
        gis = list(self.m.global_index[li] for li in lis)
        expected = list(range(16, 39))
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = 25
        self.assertRaises(IndexError, self.m.global_index.__getitem__, li)


class TestCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_local_index(self):
        gis = (2, 6, 10, 14)
        lis = tuple(self.m.local_index[gi] for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_local_index_KeyError(self):
        gi = 3
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

        gi = 7
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

    def test_global_index(self):
        lis = range(4)
        gis = tuple(self.m.global_index[li] for li in lis)
        expected = (2, 6, 10, 14)
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_index.__getitem__, li)


class TestBlockCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4,
                       block_size=2)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_local_index(self):
        """Test the local_index method of BlockCyclicMap."""
        gis = (2, 3, 10, 11)
        lis = tuple(self.m.local_index[gi] for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_local_index_KeyError(self):
        gi = 4
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

        gi = 12
        self.assertRaises(KeyError, self.m.local_index.__getitem__, gi)

    def test_global_index(self):
        lis = range(4)
        gis = tuple(self.m.global_index[li] for li in lis)
        expected = (2, 3, 10, 11)
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_index.__getitem__, li)


class TestMapEquivalences(unittest.TestCase):

    def test_compare_bcm_bm_local_index(self):
        """Test Block-Cyclic against Block map."""
        start = 4
        size = 16
        grid = 4
        block = size // grid
        dimdict = dict(start=start, size=size, proc_grid_size=grid)

        bcm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                              [('dist_type', 'c'),
                                               ('block_size', block)]))
        bm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                             [('dist_type', 'b'),
                                              ('stop', size // grid +
                                                       start)]))
        bcm_lis = [bcm.local_index[e] for e in range(4, 8)]
        bm_lis = [bm.local_index[e] for e in range(4, 8)]
        self.assertEqual(bcm_lis, bm_lis)

    def test_compare_bcm_cm_local_index(self):
        """Test Block-Cyclic against Cyclic map."""
        start = 1
        size = 16
        grid = 4
        block = 1
        dimdict = dict(start=start, size=size, proc_grid_size=grid,
                       block_size=block)
        bcm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                              [('dist_type', 'c')]))
        cm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                             [('dist_type', 'c')]))
        bcm_lis = [bcm.local_index[e] for e in range(1, 16, 4)]
        cm_lis = [cm.local_index[e] for e in range(1, 16, 4)]
        self.assertEqual(bcm_lis, cm_lis)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
