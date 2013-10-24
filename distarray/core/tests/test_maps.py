import unittest
from distarray.core import maps


class TestBlockMap(unittest.TestCase):

    def test_owner(self):
        """
        Test the owner method of BlockMap.
        """
        m = maps.BlockMap(16,4)
        owners = [m.owner(e) for e in range(16)]
        self.assertEqual(4*[0]+4*[1]+4*[2]+4*[3],owners)
        m = maps.BlockMap(17,4)
        owners = [m.owner(e) for e in range(17)]
        self.assertEqual(5*[0]+5*[1]+5*[2]+2*[3],owners)
        m = maps.BlockMap(15,4)
        owners = [m.owner(e) for e in range(15)]
        self.assertEqual(4*[0]+4*[1]+4*[2]+3*[3],owners)

    def test_local_index(self):
        """
        Test the local_index method of BlockMap.
        """
        m = maps.BlockMap(16,4)
        p = [m.local_index(i) for i in range(16)]
        self.assertEqual(4*list(range(4)),p)
        m = maps.BlockMap(17,4)
        p = [m.local_index(i) for i in range(17)]
        self.assertEqual(3*list(range(5))+[0,1],p)
        m = maps.BlockMap(15,4)
        p = [m.local_index(i) for i in range(15)]
        self.assertEqual(3*list(range(4))+[0,1,2],p)
        m = maps.BlockMap(10,2)
        p = [m.local_index(i) for i in range(10)]
        self.assertEqual(2*list(range(5)),p)


class TestCyclicMap(unittest.TestCase):

    def test_owner(self):
        """
        Test the owner method of CyclicMap.
        """
        m = maps.CyclicMap(16,4)
        owners = [m.owner(e) for e in range(16)]
        self.assertEqual(4*list(range(4)),owners)
        m = maps.CyclicMap(17,4)
        owners = [m.owner(e) for e in range(17)]
        self.assertEqual(4*list(range(4))+[0],owners)
        m = maps.CyclicMap(15,4)
        owners = [m.owner(e) for e in range(15)]
        self.assertEqual(3*list(range(4))+[0,1,2],owners)

    def test_local_index(self):
        """
        Test the local_index method of CyclicMap.
        """
        m = maps.CyclicMap(16,4)
        p = [m.local_index(i) for i in range(16)]
        self.assertEqual(4*[0]+4*[1]+4*[2]+4*[3],p)
        m = maps.CyclicMap(17,4)
        p = [m.local_index(i) for i in range(17)]
        self.assertEqual(4*[0]+4*[1]+4*[2]+4*[3]+[4],p)
        m = maps.CyclicMap(15,4)
        p = [m.local_index(i) for i in range(15)]
        self.assertEqual(4*[0]+4*[1]+4*[2]+3*[3],p)
        m = maps.BlockMap(10,2)
        p = [m.local_index(i) for i in range(10)]
        self.assertEqual(2*list(range(5)),p)


class TestBlockCyclicMap(unittest.TestCase):

    def test_owner(self):
        """
        Test the owner method of CyclicMap.
        """
        size = 20
        grid = 5
        block = 2
        bcm = maps.BlockCyclicMap(size,grid,block)
        owners = [bcm.owner(e) for e in range(size)]
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2
        self.assertEqual(expected,owners)

    def test_compare_bcm_bm_owner(self):
        """Test Block-Cyclic against Block map."""
        size = 17
        grid = 3
        block = (size // grid) + 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        bm = maps.BlockMap(size,grid)
        bcm_owners = [bcm.owner(e) for e in range(size)]
        bm_owners = [bm.owner(e) for e in range(size)]
        self.assertEqual(bcm_owners,bm_owners)

    def test_compare_bcm_cm_owner(self):
        """Test Block-Cyclic against Cyclic map."""
        size = 23
        grid = 7
        block = 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        cm = maps.CyclicMap(size,grid)
        bcm_owners = [bcm.owner(e) for e in range(size)]
        cm_owners = [cm.owner(e) for e in range(size)]
        self.assertEqual(bcm_owners,cm_owners)

    def test_local_index(self):
        """
        Test the local_index method of BlockCyclicMap.
        """
        size = 20
        grid = 5
        block = 2
        bcm = maps.BlockCyclicMap(size,grid,block)
        lindices = [bcm.local_index(i) for i in range(size)]
        expected = ([0, 1] * grid) + ([2, 3] * grid)
        self.assertEqual(lindices, expected)

    def test_compare_bcm_bm_local_index(self):
        """Test Block-Cyclic against Block map."""
        size = 17
        grid = 3
        block = (size // grid) + 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        bm = maps.BlockMap(size,grid)
        bcm_lindices = [bcm.local_index(e) for e in range(size)]
        bm_lindices = [bm.local_index(e) for e in range(size)]
        self.assertEqual(bcm_lindices,bm_lindices)

    def test_compare_bcm_cm_local_index(self):
        """Test Block-Cyclic against Cyclic map."""
        size = 23
        grid = 7
        block = 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        cm = maps.CyclicMap(size,grid)
        bcm_lindices = [bcm.local_index(e) for e in range(size)]
        cm_lindices = [cm.local_index(e) for e in range(size)]
        self.assertEqual(bcm_lindices,cm_lindices)

    def test_global_index(self):
        """
        Test the local_index method of BlockCyclicMap.
        """
        size = 20
        grid = 5
        block = 2
        bcm = maps.BlockCyclicMap(size,grid,block)
        gindices = [bcm.global_index(o, p) for o in range(grid) for p in
                    range(size//grid)]
        expected = [0, 1, 10, 11, 2, 3, 12, 13, 4, 5, 14, 15, 6,
                    7, 16, 17, 8, 9, 18, 19]
        self.assertEqual(gindices, expected)

    def test_compare_bcm_bm_global_index(self):
        """Test Block-Cyclic against Block map."""
        size = 17
        grid = 3
        block = (size // grid) + 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        bm = maps.BlockMap(size,grid)
        bcm_gindices = [bcm.global_index(o, p) for o in range(grid) for p in
                        range(size//grid)]
        bm_gindices = [bm.global_index(o, p) for o in range(grid) for p in
                       range(size//grid)]
        self.assertEqual(bcm_gindices,bm_gindices)

    def test_compare_bcm_cm_global_index(self):
        """Test Block-Cyclic against Cyclic map."""
        size = 23
        grid = 7
        block = 1
        bcm = maps.BlockCyclicMap(size,grid,block)
        cm = maps.CyclicMap(size,grid)
        bcm_gindices = [bcm.global_index(o, p) for o in range(grid) for p in
                        range(size//grid)]
        cm_gindices = [cm.global_index(o, p) for o in range(grid) for p in
                       range(size//grid)]
        self.assertEqual(bcm_gindices,cm_gindices)


class TestRegistry(unittest.TestCase):

    def test_get_class(self):
        """
        Test getting map classes by string identifier.
        """
        mc = maps.get_map_class('b')
        self.assertEqual(mc,maps.BlockMap)
        mc = maps.get_map_class('c')
        self.assertEqual(mc,maps.CyclicMap)
        mc = maps.get_map_class('bc')
        self.assertEqual(mc,maps.BlockCyclicMap)

    def test_get_class_pass(self):
        """
        Test getting a map class by the class itself.
        """
        mc = maps.get_map_class(maps.BlockMap)
        self.assertEqual(mc, maps.BlockMap)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
