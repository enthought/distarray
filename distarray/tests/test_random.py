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

    def test_seed_same(self):
        """ Test that the same seed generates the same sequence. """
        shape = (8, 6)
        seed = 0xfeedbeef
        make_distinct = True
        # Seed and get some random numbers.
        self.random.seed(seed, make_distinct=make_distinct)
        a = self.random.rand(shape)
        aa = a.toarray()
        print aa
        # Seed again and get more random numbers.
        self.random.seed(seed, make_distinct=make_distinct)
        b = self.random.rand(shape)
        bb = b.toarray()
        print bb
        # For an explicit seed, these should match exactly.
        self.assertTrue((aa == bb).all())

    def test_seed_none(self):
        """ Test that if seed=None, the sequences are not deterministic. """
        shape = (8, 6)
        # Seed and get some random numbers.
        self.random.seed(None, make_distinct=True)
        a = self.random.rand(shape)
        aa = a.toarray()
        print aa
        # Seed again and get more random numbers.
        self.random.seed(None, make_distinct=True)
        b = self.random.rand(shape)
        bb = b.toarray()
        print bb
        # For seed=None, these should *not* match.
        self.assertFalse((aa == bb).all())

    def get_rand_array_per_engine(self, num_cols):
        """ Get a distarray of random numbers,
        with each row coming from a separate engine.
        """
        num_engines = len(self.context.targets)
        shape = (num_engines, num_cols)
        darr = self.random.rand(size=shape, dist={0: 'c', 1: 'n'})
        return darr

    def test_rand_per_engine_distinct(self):
        """ Test that, if make_distinct=True when seeding,
        that each engine produces a different sequence of random numbers,
        while if make_distinct=False, then each engine produces
        the same sequence.
        """
        # Seed generators so that each engine is different.
        seed = [0x12345678, 0xdeadbeef, 42]
        self.random.seed(seed, make_distinct=True)
        # Get array of random values, with one row per engine.
        a = self.get_rand_array_per_engine(6)
        aa = a.toarray()
        # Each row should be different. We just test consecutive rows.
        num_rows = aa.shape[0]
        for r in range(num_rows - 1):
            r0 = aa[r, :]
            r1 = aa[r + 1, :]
            self.assertFalse((r0 == r1).all())
        # Now seed so that each engine is the same, and get another array.
        self.random.seed(seed, make_distinct=False)
        a = self.get_rand_array_per_engine(6)
        aa = a.toarray()
        # Now each row should be the same. We just test consecutive rows.
        num_rows = aa.shape[0]
        for r in range(num_rows - 1):
            r0 = aa[r, :]
            r1 = aa[r + 1, :]
            self.assertTrue((r0 == r1).all())

    def XXXtest_get_states(self):
        r = self.random.get_states()
        print 'result:'
        print r


if __name__ == '__main__':
    unittest.main(verbosity=2)
