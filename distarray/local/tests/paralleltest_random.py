# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
import numpy as np

from distarray.local import random as local_random
from distarray.testing import MpiTestCase


class TestBasic(MpiTestCase):
    """Run basic shape/size tests on functions in `random.py`."""

    def shape_asserts(self, la):
        self.assertEqual(la.global_shape, (16, 16))
        self.assertEqual(la.dist, ('b', 'n'))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(la.base_comm, self.comm)
        self.assertEqual(la.comm_size, 4)
        self.assertTrue(la.comm_rank in range(4))
        self.assertEqual(la.comm.Get_topo(),
                         (list(la.grid_shape),
                          [0, 0], list(la.cart_coords)))
        self.assertEqual(len(la.maps), 2)
        self.assertEqual(la.global_shape, (16, 16))
        self.assertEqual(la.grid_shape, (4, 1))
        self.assertEqual(la.local_shape, (4, 16))
        self.assertEqual(la.local_array.shape, la.local_shape)
        self.assertEqual(la.local_array.dtype, la.dtype)

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
        la = local_random.beta(2, 5, size=(16, 16), grid_shape=(4, 1), comm=self.comm)
        self.shape_asserts(la)

    def test_normal(self):
        la = local_random.normal(size=(16, 16), grid_shape=(4, 1), comm=self.comm)
        self.shape_asserts(la)

    def test_rand(self):
        la = local_random.rand(size=(16, 16), grid_shape=(4, 1), comm=self.comm)
        self.shape_asserts(la)

    def test_randint(self):
        la = local_random.randint(0, 10, size=(16, 16), grid_shape=(4, 1),
                        comm=self.comm)
        self.shape_asserts(la)

    def test_randn(self):
        la = local_random.randn((16, 16), grid_shape=(4, 1), comm=self.comm)
        self.shape_asserts(la)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
