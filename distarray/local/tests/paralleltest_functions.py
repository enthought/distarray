# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from distarray.local import arecompatible

import distarray.local.localarray as localarray
from distarray.local.localarray import LocalArray
from distarray.testing import MpiTestCase


class TestFunctions(MpiTestCase):

    def test_arecompatible(self):
        """Test if two DistArrays are compatible."""
        a = LocalArray((16,16), dtype='int64', comm=self.comm)
        b = LocalArray((16,16), dtype='float32', comm=self.comm)
        self.assertEqual(arecompatible(a,b), True)
        a = LocalArray((16, 16), dtype='int64', dist='c', comm=self.comm)
        b = LocalArray((16, 16), dtype='float32', dist='b', comm=self.comm)
        self.assertEqual(arecompatible(a,b), False)

    def test_fromfunction(self):
        """Can we build an array using fromfunction and a trivial function?"""
        def f(*global_inds):
            return 1.0

        a = localarray.fromfunction(f, (16, 16), dtype='int64',
                                    dist=('b', 'c'), comm=self.comm)
        self.assertEqual(a.global_shape, (16,16))
        self.assertEqual(a.dtype, np.dtype('int64'))
        for global_inds, value in localarray.ndenumerate(a):
            self.assertEqual(1.0, value)

    def test_fromfunction_complicated(self):
        """Can we build an array using fromfunction and a nontrivial function."""
        def f(*global_inds):
            return sum(global_inds)

        a = localarray.fromfunction(f, (16, 16), dtype='int64',
                                    dist=('b', 'c'), comm=self.comm)
        self.assertEqual(a.global_shape, (16,16))
        self.assertEqual(a.dtype, np.dtype('int64'))
        for global_inds, value in localarray.ndenumerate(a):
            self.assertEqual(sum(global_inds), value)


class TestCreationFuncs(MpiTestCase):

    def test_zeros(self):
        size = self.comm_size
        nrows = size * 3
        a = localarray.zeros((nrows, 20), comm=self.comm)
        expected = np.zeros((nrows // size, 20))
        assert_array_equal(a.local_array, expected)

    def test_ones(self):
        size = self.comm_size
        nrows = size * 3
        a = localarray.ones((nrows, 20), comm=self.comm)
        expected = np.ones((nrows // size, 20))
        assert_array_equal(a.local_array, expected)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
