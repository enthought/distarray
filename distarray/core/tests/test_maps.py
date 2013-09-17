import unittest
from distarray.core import maps


class TestMapBase(unittest.TestCase):

    def test_init(self):
        """
        Test the __init__ method of the base Map class.
        """
        m = maps.Map(16,4)
        self.assertEqual(m.local_shape,4)
        m = maps.Map(17,4)
        self.assertEqual(m.local_shape,5)
        m = maps.Map(15,4)
        self.assertEqual(m.local_shape,4)


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
