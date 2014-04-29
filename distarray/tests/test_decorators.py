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

from distarray.context import Context
from distarray.decorators import FunctionRegistrationBase
from distarray.error import ContextError


class TestFunctionRegistrationBase(TestCase):

    def test_determine_context(self):
        context = Context()
        context2 = Context()  # for cross Context checking
        da = context.ones((2, 2))

        def dummy_func(ctx, *args, **kwargs):
            def fn(x):
                return x
            db = FunctionRegistrationBase(fn, ctx)
            return db.determine_context(args, kwargs)

        self.assertEqual(dummy_func(context, 6, 7), context)
        self.assertEqual(dummy_func(context, 'ab', da), context)
        self.assertEqual(dummy_func(context, a=da), context)
        self.assertEqual(dummy_func(context, a=da), context)

        db = context2.ones((2, 2))
        self.assertRaises(ContextError, dummy_func, context, db)

    def test_key_and_push_args(self):
        context = Context()

        da = context.ones((2, 2))
        db = da*2

        def dummy_func(ctx, *args, **kwargs):
            def fn(x):
                return x
            db = FunctionRegistrationBase(fn, ctx)
            return db.build_args(args, kwargs)

        # Push some distarrays
        arg_keys1, kw_keys1 = dummy_func(context, da, db, foo=da, bar=db)
        # with some other data too
        arg_keys2, kw_keys2 = dummy_func(context, da, 'question', answer=42, foo=db)

        self.assertSequenceEqual(arg_keys1, (da.key, db.key))
        # assert we pushed the right key, keystr pair
        self.assertDictEqual({'foo': da.key,  'bar': db.key}, kw_keys1)

        # lots of string manipulation to parse out the relevant pieces
        # of the python commands.
        self.assertSequenceEqual(arg_keys2, (da.key, 'question'))

        self.assertSetEqual(set("answer foo".split()), set(kw_keys2.keys()))
        self.assertIn(db.key, kw_keys2.values())


class TestLocalDecorator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = ctx = Context()
        cls.da = cls.context.empty((5, 5))
        cls.da.fill(2 * numpy.pi)

        def local_add50(da):
            return da + 50
        ctx.localize(local_add50)

        def local_add_num(da, num):
            return da + num
        ctx.localize(local_add_num)

        def assert_allclose(da, db):
            assert numpy.allclose(da, db), "Arrays not equal within tolerance."
        ctx.localize(assert_allclose)

        def local_sin(da):
            return numpy.sin(da)
        ctx.localize(local_sin)

        def local_sum(da):
            return numpy.sum(da.get_localarray())
        ctx.localize(local_sum)

        def call_barrier(da):
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
            return da
        ctx.localize(call_barrier)

        def local_add_nums(da, num1, num2, num3):
            return da + num1 + num2 + num3
        ctx.localize(local_add_nums)

        def local_add_distarrayproxies(da, dg):
            return da + dg
        ctx.localize(local_add_distarrayproxies)

        def local_add_mixed(da, num1, dg, num2):
            return da + num1 + dg + num2
        ctx.localize(local_add_mixed)

        def local_add_ndarray(da, num, ndarr):
            return da + num + ndarr
        ctx.localize(local_add_ndarray)

        def local_add_kwargs(da, num1, num2=55):
            return da + num1 + num2
        ctx.localize(local_add_kwargs)

        def local_add_supermix(da, num1, db, num2, dc, num3=99, num4=66):
            return da + num1 + db + num2 + dc + num3 + num4
        ctx.localize(local_add_supermix)

        def local_none(da):
            return None
        ctx.localize(local_none)

        def parameterless():
            """This is a parameterless function."""
            return None
        ctx.localize(parameterless)

    @classmethod
    def tearDownClass(cls):
        cls.context.cleanup()

    def test_local(self):
        context = Context()

        """Test the @local decorator"""
        da = context.empty((4, 4))
        a = numpy.empty((4, 4))

        def fill_a(a):
            for (i, j), _ in numpy.ndenumerate(a):
                a[i, j] = i + j
            return a

        def fill_da(da):
            for i in da.maps[0].global_iter:
                for j in da.maps[1].global_iter:
                    da.global_index[i, j] = i + j
            return da
        context.localize(fill_da)

        da = context.fill_da(da)
        a = fill_a(a)

        assert_array_equal(da.toarray(), a)

    def test_local_sin(self):
        db = self.context.local_sin(self.da)
        self.context.assert_allclose(db, 0)

    def test_local_add50(self):
        dc = self.context.local_add50(self.da)
        self.context.assert_allclose(dc, 2 * numpy.pi + 50)

    def test_local_sum(self):
        dd = self.context.local_sum(self.da)
        lshapes = self.da.get_localshapes()
        expected = []
        for lshape in lshapes:
            expected.append(lshape[0] * lshape[1] * (2 * numpy.pi))
        for (v, e) in zip(dd, expected):
            self.assertAlmostEqual(v, e, places=5)

    def test_local_add_num(self):
        de = self.context.local_add_num(self.da, 11)
        self.context.assert_allclose(de, 2 * numpy.pi + 11)

    def test_local_add_nums(self):
        df = self.context.local_add_nums(self.da, 11, 12, 13)
        self.context.assert_allclose(df, 2 * numpy.pi + 11 + 12 + 13)

    def test_local_add_distarrayproxies(self):
        dg = self.context.empty((5, 5))
        dg.fill(33)
        dh = self.context.local_add_distarrayproxies(self.da, dg)
        self.context.assert_allclose(dh, 33 + 2 * numpy.pi)

    def test_local_add_mixed(self):
        di = self.context.empty((5, 5))
        di.fill(33)
        dj = self.context.local_add_mixed(self.da, 11, di, 12)
        self.context.assert_allclose(dj, 2 * numpy.pi + 11 + 33 + 12)

    @unittest.skip('Locally adding ndarrays not supported.')
    def test_local_add_ndarray(self):
        shp = self.da.get_localshapes()[0]
        ndarr = numpy.empty(shp)
        ndarr.fill(33)
        dk = self.context.local_add_ndarray(self.da, 11, ndarr)
        self.context.assert_allclose(dk, 2 * numpy.pi + 11 + 33)

    def test_local_add_kwargs(self):
        dl = self.context.local_add_kwargs(self.da, 11, num2=12)
        self.context.assert_allclose(dl, 2 * numpy.pi + 11 + 12)

    def test_local_add_supermix(self):
        dm = self.context.empty((5, 5))
        dm.fill(22)
        dn = self.context.empty((5, 5))
        dn.fill(44)
        do = self.context.local_add_supermix(self.da, 11, dm, 33, dc=dn, num3=55)
        expected = 2 * numpy.pi + 11 + 22 + 33 + 44 + 55 + 66
        self.context.assert_allclose(do, expected)

    def test_local_none(self):
        dp = self.context.local_none(self.da)
        self.assertTrue(dp is None)

    def test_barrier(self):
        self.context.call_barrier(self.da)

    def test_parameterless(self):
        result = self.context.parameterless()
        self.assertIsNone(result)

    def test_function_metadata(self):
        name = "parameterless"
        docstring = """This is a parameterless function."""
        self.assertEqual(self.context.parameterless.__name__, name)
        self.assertEqual(self.context.parameterless.__doc__, docstring)


class TestVectorizeDecorator(TestCase):

    def test_vectorize(self):
        """Test the @vectorize decorator for parity with NumPy's"""

        ctx = Context()

        a = numpy.arange(16).reshape(4, 4)
        da = ctx.fromndarray(a)

        def da_fn(a, b, c):
            return a**2 + b + c
        ctx.vectorize(da_fn)

        @numpy.vectorize
        def a_fn(a, b, c):
            return a**2 + b + c

        a = a_fn(a, a, 6)
        db = ctx.da_fn(da, da, 6)
        assert_array_equal(db.toarray(), a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
