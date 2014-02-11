import unittest
from distarray.remote import maps

from six.moves import range


class TestBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(disttype='b', start=16, stop=39)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_remote_index(self):
        gis = range(16, 39)
        lis = list(self.m.remote_index[gi] for gi in gis)
        expected = list(range(23))
        self.assertEqual(lis, expected)

    def test_remote_index_KeyError(self):
        gi = 15
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

        gi = 39
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

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
        dimdict = dict(disttype='c', start=2, datasize=16, gridsize=4)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_remote_index(self):
        gis = (2, 6, 10, 14)
        lis = tuple(self.m.remote_index[gi] for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_remote_index_KeyError(self):
        gi = 3
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

        gi = 7
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

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
        dimdict = dict(disttype='bc', start=2, datasize=16, gridsize=4,
                       blocksize=2)
        self.m = maps.IndexMap.from_dimdict(dimdict)

    def test_remote_index(self):
        """Test the remote_index method of BlockCyclicMap."""
        gis = (2, 3, 10, 11)
        lis = tuple(self.m.remote_index[gi] for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_remote_index_KeyError(self):
        gi = 4
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

        gi = 12
        self.assertRaises(KeyError, self.m.remote_index.__getitem__, gi)

    def test_global_index(self):
        lis = range(4)
        gis = tuple(self.m.global_index[li] for li in lis)
        expected = (2, 3, 10, 11)
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = 5
        self.assertRaises(IndexError, self.m.global_index.__getitem__, li)


class TestMapEquivalences(unittest.TestCase):

    def test_compare_bcm_bm_remote_index(self):
        """Test Block-Cyclic against Block map."""
        start = 4
        size = 16
        grid = 4
        block = size // grid
        dimdict = dict(start=start, datasize=size, gridsize=grid)

        bcm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                              [('disttype', 'bc'),
                                               ('blocksize', block)]))
        bm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                             [('disttype', 'b'),
                                              ('stop', size // grid +
                                                       start)]))
        bcm_lis = [bcm.remote_index[e] for e in range(4, 8)]
        bm_lis = [bm.remote_index[e] for e in range(4, 8)]
        self.assertEqual(bcm_lis, bm_lis)

    def test_compare_bcm_cm_remote_index(self):
        """Test Block-Cyclic against Cyclic map."""
        start = 1
        size = 16
        grid = 4
        block = 1
        dimdict = dict(start=start, datasize=size, gridsize=grid,
                       blocksize=block)
        bcm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                              [('disttype', 'bc')]))
        cm = maps.IndexMap.from_dimdict(dict(list(dimdict.items()) +
                                             [('disttype', 'c')]))
        bcm_lis = [bcm.remote_index[e] for e in range(1, 16, 4)]
        cm_lis = [cm.remote_index[e] for e in range(1, 16, 4)]
        self.assertEqual(bcm_lis, cm_lis)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
