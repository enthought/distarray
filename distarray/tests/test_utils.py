import unittest
from distarray import utils


class TestMultPartitions(unittest.TestCase):
    """
    Test the multiplicative parition code.
    """

    def test_both_methods(self):
        """
        Do the two methods of computing the multiplicative partitions agree?
        """
        for s in [2, 3]:
            for n in range(2, 512):
                self.assertEqual(utils.mult_partitions(n, s),
                                 utils.create_factors(n, s))


if __name__ == '__main__':
    unittest.main(verbosity=2)
