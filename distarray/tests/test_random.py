# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""Test random.py"""

import unittest
from numpy.testing import assert_array_equal

from distarray.context import Context
from distarray.random import Random


class TestRandom(unittest.TestCase):
    """Test the Random classes methods, since the expected results are
    'random' we just check for the correct shape. Which is dumb, but
    better than nothing.
    """

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.random = Random(cls.context)

    @classmethod
    def tearDownClass(cls):
        """Close the client connections"""
        client = cls.context.view.client
        del cls.random
        del cls.context
        client.close()

    def test_rand(self):
        shape = (3, 4)
        a = self.random.rand(shape)
        self.assertEqual(a.shape, shape)

    def test_normal(self):
        size = (3, 4)  # aka shape
        a = self.random.normal(size=size)
        self.assertEqual(a.shape, size)

    def test_randint(self):
        low = 1
        size = (3, 4)  # aka shape
        a = self.random.randint(low, size=size)
        self.assertEqual(a.shape, size)

    def test_randn(self):
        shape = (3, 4)
        a = self.random.randn(shape)
        self.assertEqual(a.shape, shape)

    def test_seed(self):
        """ Test that the same seed generates the same sequence. """
        shape = (8, 6)
        seed_use = 0xfeedbeef
        make_distinct = True   #False
        # Seed once and get some random numbers.
        self.random.seed(seed_use, make_distinct=make_distinct)
        a = self.random.rand(shape)
        aa = a.toarray()
        print aa
        # Reseed with same seed, results should be the same.
        self.random.seed(seed_use, make_distinct=make_distinct)
        b = self.random.rand(shape)
        bb = b.toarray()
        print bb
        assert_array_equal(aa, bb)

    def test_get_states(self):
        r = self.random.get_states()
        print 'result:'
        print r


if __name__ == '__main__':
    unittest.main(verbosity=2)
