# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Test decorators, these need a separate module because we are defining
functions statically, but decorators need context objects which are
defined dynamically. To remedy this, we define a global Context, and use
it throughout this module.
"""

import unittest
from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from distarray.testing import ContextTestCase, check_targets
from distarray.dist.context import Context
from distarray.dist.maps import Distribution
from distarray.dist.decorators import DecoratorBase, local, vectorize
from distarray.error import DistributionError


class TestDecoratorBase(TestCase):

    def test_determine_distribution(self):
        context = Context()
        context2 = Context()  # for cross Context checking
        dist = Distribution(context, (2, 2))
        dist2 = Distribution(context2, (2, 2))
        da = context.ones(dist)

        def dummy_func(*args, **kwargs):
            fn = lambda x: x
            db = DecoratorBase(fn)
            return db.determine_distribution(args, kwargs)

        self.assertEqual(dummy_func(6, 7, dist), dist)
        self.assertEqual(dummy_func('ab', da), dist)
        self.assertEqual(dummy_func(a=da), dist)
        self.assertEqual(dummy_func(dist, a=da), dist)

        self.assertRaises(TypeError, dummy_func, 'foo')
        self.assertRaises(DistributionError, dummy_func, dist, dist2)

    def test_key_and_push_args(self):
        context = Context()

        distribution = Distribution(context, (2, 2))
        da = context.ones(distribution)
        db = da*2

        def dummy_func(*args, **kwargs):
            fn = lambda x: x
            db = DecoratorBase(fn)
            return db.key_and_push_args(args, kwargs)

        # Push some distarrays
        arg_keys1, kw_keys1 = dummy_func(da, db, foo=da, bar=db)
        # with some other data too
        arg_keys2, kw_keys2 = dummy_func(da, 'question', answer=42, foo=db)

        self.assertEqual(arg_keys1, "(%s, %s,)" % (da.key, db.key))
        # assert we pushed the right key, keystr pair
        self.assertTrue("'foo': %s" % (da.key) in kw_keys1)
        self.assertTrue("'bar': %s" % (db.key) in kw_keys1)

        # lots of string manipulation to parse out the relevant pieces
        # of the python commands.
        self.assertEqual(arg_keys2[1: -2].split(', ')[0], da.key)

        _key = arg_keys2[1: -2].split(', ')[1]
        self.assertEqual(context._pull(_key, targets=[0])[0], 'question')
        self.assertTrue("'answer'" in kw_keys2)

        self.assertTrue("'foo'" in kw_keys2)
        self.assertTrue(db.key in kw_keys2)


class TestLocalDecorator(ContextTestCase):

    ntargets = 'any'

    # Functions for @local decorator tests. These are here so we can
    # guarantee they are pushed to the engines before we try to use them.
    @local
    def local_add50(da):
        return da + 50

    @local
    def local_add_num(da, num):
        return da + num

    @local
    def assert_allclose(da, db):
        assert numpy.allclose(da, db), "Arrays not equal within tolerance."

    @local
    def local_sin(da):
        return numpy.sin(da)

    @local
    def local_sum(da):
        return numpy.sum(da.ndarray)

    @local
    def call_barrier(da):
        da.comm.Barrier()
        return da

    @local
    def local_add_nums(da, num1, num2, num3):
        return da + num1 + num2 + num3

    @local
    def local_add_distarrayproxies(da, dg):
        return da + dg

    @local
    def local_add_mixed(da, num1, dg, num2):
        return da + num1 + dg + num2

    @local
    def local_add_ndarray(da, num, ndarr):
        return da + num + ndarr

    @local
    def local_add_kwargs(da, num1, num2=55):
        return da + num1 + num2

    @local
    def local_add_supermix(da, num1, db, num2, dc, num3=99, num4=66):
        return da + num1 + db + num2 + dc + num3 + num4

    @local
    def local_none(da):
        return None

    @local
    def parameterless():
        """This is a parameterless function."""
        return None

    @classmethod
    def setUpClass(cls):
        super(TestLocalDecorator, cls).setUpClass()
        distribution = Distribution(cls.context, (5, 5))
        cls.da = cls.context.empty(distribution)
        cls.da.fill(2 * numpy.pi)

    def test_local(self):
        """Test the @local decorator"""
        context = Context()

        distribution = Distribution(context, (4, 4))
        da = context.empty(distribution)
        a = numpy.empty((4, 4))

        def fill_a(a):
            for (i, j), _ in numpy.ndenumerate(a):
                a[i, j] = i + j
            return a

        @local
        def fill_da(da):
            for i in da.distribution[0].global_iter:
                for j in da.distribution[1].global_iter:
                    da.global_index[i, j] = i + j
            return da

        da = fill_da(da)
        a = fill_a(a)

        assert_array_equal(da.toarray(), a)

    def test_different_contexts(self):
        check_targets(required=4, available=len(self.context.targets))

        ctx1 = Context(targets=range(4))
        ctx2 = Context(targets=range(3))
        distribution1 = Distribution(ctx1, (10,))
        distribution2 = Distribution(ctx2, (10,))
        da1 = ctx1.ones(distribution1)
        da2 = ctx2.ones(distribution2)
        db1 = self.local_sin(da1)
        db2 = self.local_sin(da2)
        ndarr1 = db1.toarray()
        ndarr2 = db2.toarray()
        assert_array_equal(ndarr1, ndarr2)

    def test_local_sin(self):
        db = self.local_sin(self.da)
        self.assert_allclose(db, 0)

    def test_local_add50(self):
        dc = self.local_add50(self.da)
        self.assert_allclose(dc, 2 * numpy.pi + 50)

    def test_local_sum(self):
        dd = self.local_sum(self.da)
        if self.ntargets == 1:
            dd = [dd]
        lshapes = self.da.localshapes()
        expected = []
        for lshape in lshapes:
            expected.append(lshape[0] * lshape[1] * (2 * numpy.pi))
        for (v, e) in zip(dd, expected):
            self.assertAlmostEqual(v, e, places=5)

    def test_local_add_num(self):
        de = self.local_add_num(self.da, 11)
        self.assert_allclose(de, 2 * numpy.pi + 11)

    def test_local_add_nums(self):
        df = self.local_add_nums(self.da, 11, 12, 13)
        self.assert_allclose(df, 2 * numpy.pi + 11 + 12 + 13)

    def test_local_add_distarrayproxies(self):
        dg = self.context.empty(self.da.distribution)
        dg.fill(33)
        dh = self.local_add_distarrayproxies(self.da, dg)
        self.assert_allclose(dh, 33 + 2 * numpy.pi)

    def test_local_add_mixed(self):
        di = self.context.empty(self.da.distribution)
        di.fill(33)
        dj = self.local_add_mixed(self.da, 11, di, 12)
        self.assert_allclose(dj, 2 * numpy.pi + 11 + 33 + 12)

    @unittest.skip('Locally adding ndarrays not supported.')
    def test_local_add_ndarray(self):
        shp = self.da.localshapes()[0]
        ndarr = numpy.empty(shp)
        ndarr.fill(33)
        dk = self.local_add_ndarray(self.da, 11, ndarr)
        self.assert_allclose(dk, 2 * numpy.pi + 11 + 33)

    def test_local_add_kwargs(self):
        dl = self.local_add_kwargs(self.da, 11, num2=12)
        self.assert_allclose(dl, 2 * numpy.pi + 11 + 12)

    def test_local_add_supermix(self):
        dm = self.context.empty(self.da.distribution)
        dm.fill(22)
        dn = self.context.empty(self.da.distribution)
        dn.fill(44)
        do = self.local_add_supermix(self.da, 11, dm, 33, dc=dn, num3=55)
        expected = 2 * numpy.pi + 11 + 22 + 33 + 44 + 55 + 66
        self.assert_allclose(do, expected)

    def test_local_none(self):
        dp = self.local_none(self.da)
        self.assertTrue(dp is None)

    def test_barrier(self):
        self.call_barrier(self.da)

    def test_parameterless(self):
        self.assertRaises(TypeError, self.parameterless)

    def test_function_metadata(self):
        name = "parameterless"
        docstring = """This is a parameterless function."""
        self.assertEqual(self.parameterless.__name__, name)
        self.assertEqual(self.parameterless.__doc__, docstring)


class TestVectorizeDecorator(TestCase):

    def test_vectorize(self):
        """Test the @vectorize decorator for parity with NumPy's"""

        context = Context()

        a = numpy.arange(16).reshape(4, 4)
        da = context.fromndarray(a)

        @vectorize
        def da_fn(a, b, c):
            return a**2 + b + c

        @numpy.vectorize
        def a_fn(a, b, c):
            return a**2 + b + c

        a = a_fn(a, a, 6)
        db = da_fn(da, da, 6)
        assert_array_equal(db.toarray(), a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
