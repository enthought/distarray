import unittest
from distarray.local import maps

from six.moves import range


class TestBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(start=16, stop=39)
        self.m = maps.BlockMap(dimdict)

    def test_local_index(self):
        gis = range(16, 39)
        lis = list(self.m.local_index(gi) for gi in gis)
        expected = list(range(23))
        self.assertEqual(lis, expected)

    def test_local_index_IndexError(self):
        gi = 15
        self.assertRaises(IndexError, self.m.local_index, gi)

        gi = 39
        self.assertRaises(IndexError, self.m.local_index, gi)

    def test_global_index(self):
        lis = range(23)
        gis = list(self.m.global_index(li) for li in lis)
        expected = list(range(16, 39))
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = -1
        self.assertRaises(IndexError, self.m.global_index, li)

        li = 25
        self.assertRaises(IndexError, self.m.global_index, li)


class TestRegularBlockMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(gridrank=2, datasize=16, gridsize=4)
        self.m = maps.RegularBlockMap(dimdict)

    def test_local_index(self):
        gis = range(8, 12)
        lis = list(self.m.local_index(gi) for gi in gis)
        expected = list(range(4))
        self.assertEqual(lis, expected)

    def test_local_index_IndexError(self):
        gi = 7
        self.assertRaises(IndexError, self.m.local_index, gi)

        gi = 12
        self.assertRaises(IndexError, self.m.local_index, gi)

    def test_global_index(self):
        lis = range(4)
        gis = list(self.m.global_index(li) for li in lis)
        expected = list(range(8, 12))
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = -1
        self.assertRaises(IndexError, self.m.global_index, li)

        li = 5
        self.assertRaises(IndexError, self.m.global_index, li)


class TestRegularCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(gridrank=2, datasize=16, gridsize=4)
        self.m = maps.RegularCyclicMap(dimdict)

    def test_local_index(self):
        gis = (2, 6, 10, 14)
        lis = tuple(self.m.local_index(gi) for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_local_index_IndexError(self):
        gi = 3
        self.assertRaises(IndexError, self.m.local_index, gi)

        gi = 7
        self.assertRaises(IndexError, self.m.local_index, gi)

    def test_global_index(self):
        lis = range(4)
        gis = tuple(self.m.global_index(li) for li in lis)
        expected = (2, 6, 10, 14)
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = -1
        self.assertRaises(IndexError, self.m.global_index, li)

        li = 5
        self.assertRaises(IndexError, self.m.global_index, li)


class TestRegularBlockCyclicMap(unittest.TestCase):

    def setUp(self):
        dimdict = dict(gridrank=1, datasize=16, gridsize=4, blocksize=2)
        self.m = maps.RegularBlockCyclicMap(dimdict)

    def test_local_index(self):
        """Test the local_index method of BlockCyclicMap."""
        gis = (2, 3, 10, 11)
        lis = tuple(self.m.local_index(gi) for gi in gis)
        expected = tuple(range(4))
        self.assertEqual(lis, expected)

    def test_local_index_IndexError(self):
        gi = 4
        self.assertRaises(IndexError, self.m.local_index, gi)

        gi = 12
        self.assertRaises(IndexError, self.m.local_index, gi)

    def test_global_index(self):
        lis = range(4)
        gis = tuple(self.m.global_index(li) for li in lis)
        expected = (2, 3, 10, 11)
        self.assertEqual(gis, expected)

    def test_global_index_IndexError(self):
        li = -1
        self.assertRaises(IndexError, self.m.global_index, li)

        li = 5
        self.assertRaises(IndexError, self.m.global_index, li)


class TestMapEquivalences(unittest.TestCase):

    def test_compare_bcm_bm_local_index(self):
        """Test Block-Cyclic against Block map."""
        rank = 1
        size = 16
        grid = 4
        block = size // grid
        dimdict = dict(gridrank=rank, datasize=size, gridsize=grid,
                       blocksize=block)

        bcm = maps.RegularBlockCyclicMap(dimdict)
        bm = maps.RegularBlockMap(dimdict)
        bcm_lis = [bcm.local_index(e) for e in range(4, 8)]
        bm_lis = [bm.local_index(e) for e in range(4, 8)]
        self.assertEqual(bcm_lis, bm_lis)

    def test_compare_bcm_cm_local_index(self):
        """Test Block-Cyclic against Cyclic map."""
        rank = 1
        size = 16
        grid = 4
        block = 1
        dimdict = dict(gridrank=rank, datasize=size, gridsize=grid,
                       blocksize=block)
        bcm = maps.RegularBlockCyclicMap(dimdict)
        cm = maps.RegularCyclicMap(dimdict)
        bcm_lis = [bcm.local_index(e) for e in range(1, 16, 4)]
        cm_lis = [cm.local_index(e) for e in range(1, 16, 4)]
        self.assertEqual(bcm_lis, cm_lis)


class TestRegistry(unittest.TestCase):

    def test_get_class(self):
        """Test getting map classes by string identifier."""
        mc = maps.get_map_class('b')
        self.assertEqual(mc,maps.RegularBlockMap)
        mc = maps.get_map_class('c')
        self.assertEqual(mc,maps.RegularCyclicMap)
        mc = maps.get_map_class('bc')
        self.assertEqual(mc,maps.RegularBlockCyclicMap)

    def test_get_class_pass(self):
        """Test getting a map class by the class itself."""
        mc = maps.get_map_class(maps.RegularBlockMap)
        self.assertEqual(mc, maps.RegularBlockMap)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
