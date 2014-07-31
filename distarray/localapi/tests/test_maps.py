# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from distarray.externals.six.moves import range

from distarray.localapi import maps


class TestNoDistMap(unittest.TestCase):

    def setUp(self):
        size = 20
        dimdict = dict(dist_type='n', size=size)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global_index(self):
        gis = range(0, 20)
        lis = [self.m.local_from_global_index(gi) for gi in gis]
        expected = list(range(20))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_index_IndexError(self):
        gi = 20
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

    def test_global_from_local_index(self):
        lis = range(20)
        gis = [self.m.global_from_local_index(li) for li in lis]
        expected = list(range(20))
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_index_IndexError(self):
        li = 20
        self.assertRaises(IndexError, self.m.global_from_local_index, li)


class TestBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='b', size=(39-16), start=16, stop=39)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global_index(self):
        gis = range(16, 39)
        lis = [self.m.local_from_global_index(gi) for gi in gis]
        expected = list(range(23))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_index_IndexError(self):
        gi = 15
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

        gi = 39
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

    def test_global_from_local_index(self):
        lis = range(23)
        gis = [self.m.global_from_local_index(li) for li in lis]
        expected = list(range(16, 39))
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_index_IndexError(self):
        li = 25
        self.assertRaises(IndexError, self.m.global_from_local_index, li)


class TestCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4, proc_grid_rank=2)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global_index(self):
        gis = (2, 6, 10, 14)
        lis = [self.m.local_from_global_index(gi) for gi in gis]
        expected = tuple(range(4))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_index_IndexError(self):
        gi = 3
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

        gi = 7
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

    def test_global_from_local_index(self):
        lis = range(4)
        gis = [self.m.global_from_local_index(li) for li in lis]
        expected = (2, 6, 10, 14)
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_index_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_from_local_index, li)


class TestBlockCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(dist_type='c', start=2, size=16, proc_grid_size=4,
                       block_size=2)
        self.m = maps.map_from_dim_dict(dimdict)

    def test_local_from_global_index(self):
        """Test the local_index method of BlockCyclicMap."""
        gis = (2, 3, 10, 11)
        lis = [self.m.local_from_global_index(gi) for gi in gis]
        expected = tuple(range(4))
        self.assertSequenceEqual(lis, expected)

    def test_local_from_global_index_IndexError(self):
        gi = 4
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)
        gi = 12
        self.assertRaises(IndexError, self.m.local_from_global_index, gi)

    def test_global_from_local_index(self):
        lis = range(4)
        gis = [self.m.global_from_local_index(li) for li in lis]
        expected = (2, 3, 10, 11)
        self.assertSequenceEqual(gis, expected)

    def test_global_from_local_index_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_from_local_index, li)


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
        bcm_lis = [bcm.local_from_global_index(e) for e in range(4, 8)]
        bm_lis = [bm.local_from_global_index(e) for e in range(4, 8)]
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
        bcm_lis = [bcm.local_from_global_index(e) for e in range(1, 16, 4)]
        cm_lis = [cm.local_from_global_index(e) for e in range(1, 16, 4)]
        self.assertSequenceEqual(bcm_lis, cm_lis)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
