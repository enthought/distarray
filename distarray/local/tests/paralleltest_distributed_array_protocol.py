# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from distutils.version import StrictVersion

import distarray.local
from distarray.externals import six
from distarray.externals import protocol_validator
from distarray.testing import MpiTestCase, CommNullPasser


@six.add_metaclass(CommNullPasser)
class DapTestMixin(object):

    """Base test class for DAP test cases.

    Overload `setUp` and add a `self.larr` LocalArray to run this test suite
    on.
    """

    def test_with_validator(self):
        valid, msg = protocol_validator.validate(self.larr.__distarray__())
        self.assertTrue(valid, msg=msg)

    @unittest.skip("'n' -> 'b' translation breaks round trip equality.")
    def test_round_trip_equality_from_object(self):
        larr = distarray.local.LocalArray.from_distarray(self.larr,
                                                         comm=self.comm)
        self.assertEqual(larr.global_shape, self.larr.global_shape)
        self.assertEqual(larr.dist, self.larr.dist)
        self.assertEqual(larr.grid_shape, self.larr.grid_shape)
        self.assertEqual(larr.comm_size, self.larr.comm_size)
        self.assertEqual(larr.comm.Get_topo(), self.larr.comm.Get_topo())
        self.assertEqual(len(larr.maps), len(self.larr.maps))
        self.assertEqual(larr.local_shape, self.larr.local_shape)
        self.assertEqual(larr.local_array.shape, self.larr.local_array.shape)
        self.assertEqual(larr.local_array.dtype, self.larr.local_array.dtype)
        assert_array_equal(larr.local_array, self.larr.local_array)

    @unittest.skip("'n' -> 'b' translation breaks round trip equality.")
    def test_round_trip_equality_from_dict(self):
        larr = distarray.local.LocalArray.from_distarray(
            self.larr.__distarray__(), comm=self.comm)
        self.assertEqual(larr.global_shape, self.larr.global_shape)
        self.assertEqual(larr.dist, self.larr.dist)
        self.assertEqual(larr.grid_shape, self.larr.grid_shape)
        self.assertEqual(larr.comm_size, self.larr.comm_size)
        self.assertEqual(larr.comm.Get_topo(), self.larr.comm.Get_topo())
        self.assertEqual(len(larr.maps), len(self.larr.maps))
        self.assertEqual(larr.local_shape, self.larr.local_shape)
        self.assertEqual(larr.local_array.shape, self.larr.local_array.shape)
        self.assertEqual(larr.local_array.dtype, self.larr.local_array.dtype)
        assert_array_equal(larr.local_array, self.larr.local_array)

    def test_round_trip_identity(self):
        larr = distarray.local.LocalArray.from_distarray(self.larr,
                                                         comm=self.comm)
        if self.comm.Get_rank() == 0:
            idx = (0,) * larr.local_array.ndim
            larr.local_array[idx] = 99
        assert_array_equal(larr.local_array, self.larr.local_array)


class TestDapBasic(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((16, 16), grid_shape=(4, 1),
                                               comm=self.comm)


class TestDapUint(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((16, 16), dtype='uint8',
                                               grid_shape=(4, 1),
                                               comm=self.comm, buf=None)


class TestDapComplex(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((16, 16), dtype='complex128',
                                               grid_shape=(4, 1),
                                               comm=self.comm, buf=None)


class TestDapExplicitNoDist0(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((16, 16), dist={0: 'b', 1: 'n'},
                                               grid_shape=(4, 1),
                                               comm=self.comm)


class TestDapExplicitNoDist1(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((30, 60), dist={0: 'n', 1: 'b'},
                                               grid_shape=(1, 4),
                                               comm=self.comm)


class TestDapTwoDistDims(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((53, 77), dist={0: 'b', 1: 'b'},
                                               grid_shape=(2, 2),
                                               comm=self.comm)


class TestDapThreeBlockDims(DapTestMixin, MpiTestCase):

    comm_size = 12

    def setUp(self):
        self.larr = distarray.local.LocalArray((53, 77, 99),
                                               dist={0: 'b', 1: 'b', 2: 'b'},
                                               grid_shape=(2, 2, 3),
                                               comm=self.comm)


class TestDapCyclicDim(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((53, 77), dist={0: 'c'},
                                               grid_shape=(4, 1),
                                               comm=self.comm)


class TestDapCyclicBlock(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((53, 77), dist={0: 'c', 1: 'b'},
                                               grid_shape=(2, 2),
                                               comm=self.comm)


class TestDapThreeMixedDims(DapTestMixin, MpiTestCase):

    def setUp(self):
        self.larr = distarray.local.LocalArray((53, 77, 99), dtype='float64',
                                               dist={0: 'b', 1: 'n', 2: 'c'},
                                               grid_shape=(2, 1, 2),
                                               comm=self.comm)


class TestDapLopsided(DapTestMixin, MpiTestCase):

    comm_size = 2

    def setUp(self):
        if self.comm.Get_rank() == 0:
            arr = np.arange(20)
        elif self.comm.Get_rank() == 1:
            arr = np.arange(30)

        self.larr = distarray.local.LocalArray((50,), dtype='float64',
                                               dist={0: 'b', 1: 'n'},
                                               grid_shape=(2,), comm=self.comm,
                                               buf=arr)

    def test_values(self):
        if self.comm.Get_rank() == 0:
            assert_array_equal(np.arange(20), self.larr.local_array)
        elif self.comm.Get_rank() == 1:
            assert_array_equal(np.arange(30), self.larr.local_array)

        larr = distarray.local.LocalArray.from_distarray(self.larr,
                                                         comm=self.comm)
        if self.comm.Get_rank() == 0:
            assert_array_equal(np.arange(20), larr.local_array)
        elif self.comm.Get_rank() == 1:
            assert_array_equal(np.arange(30), larr.local_array)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
