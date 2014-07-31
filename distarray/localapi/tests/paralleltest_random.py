# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
import numpy as np

from distarray.localapi import random as local_random
from distarray.localapi.maps import Distribution
from distarray.testing import ParallelTestCase


class TestBasic(ParallelTestCase):
    """Run basic shape/size tests on functions in `random.py`."""

    def shape_asserts(self, la):
        self.assertEqual(la.dist, ('b', 'b'))
        self.assertEqual(la.global_shape, (16, 16))
        self.assertEqual(la.ndim, 2)
        self.assertEqual(la.global_size, 16*16)
        self.assertEqual(la.comm_size, 4)
        self.assertTrue(la.comm_rank in range(4))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(len(la.distribution), 2)
        self.assertEqual(la.global_shape, (16, 16))
        self.assertEqual(la.local_shape, (4, 16))
        self.assertEqual(la.ndarray.shape, la.local_shape)
        self.assertEqual(la.ndarray.dtype, la.dtype)

    def test_label_state(self):
        """ Test we can label the local random generator with the rank. """
        # This test is mainly intended for coverage, as the client-side
        # test of the behavior does not label the routine as covered.
        s0, orig_array, s2, s3, s4 = np.random.get_state()
        local_random.label_state(self.comm)
        s0, rank_array, s2, s3, s4 = np.random.get_state()
        # State should have changed from labeling.
        state_equal = np.all(orig_array == rank_array)
        self.assertFalse(state_equal)

    def test_beta(self):
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), grid_shape=(4, 1))
        la = local_random.beta(2, 5, distribution=d)
        self.shape_asserts(la)

    def test_normal(self):
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), grid_shape=(4, 1))
        la = local_random.normal(distribution=d)
        self.shape_asserts(la)

    def test_rand(self):
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), grid_shape=(4, 1))
        la = local_random.rand(d)
        self.shape_asserts(la)

    def test_randint(self):
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), grid_shape=(4, 1))
        la = local_random.randint(0, 10, distribution=d)
        self.shape_asserts(la)

    def test_randn(self):
        d = Distribution.from_shape(comm=self.comm,
                            shape=(16, 16), grid_shape=(4, 1))
        la = local_random.randn(distribution=d)
        self.shape_asserts(la)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
