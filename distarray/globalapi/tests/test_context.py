# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for distarray's contexts.

Many of these tests require a 4-engine cluster to be running locally.  The
engines should be launched with MPI, using the MPIEngineSetLauncher.

"""

import unittest
from random import shuffle

import numpy

from numpy.testing import assert_allclose, assert_array_equal

from distarray.testing import (DefaultContextTestCase, IPythonContextTestCase,
                               MPIContextTestCase, check_targets)
import distarray.globalapi as gapi
from distarray.globalapi.context import DistArray, Context
from distarray.globalapi.maps import Distribution
from distarray.mpionly_utils import is_solo_mpi_process
from distarray.localapi import LocalArray
from distarray.localapi.proxyize import LazyPlaceholder


@unittest.skipIf(is_solo_mpi_process(),  # not in MPI mode
                 "Cannot test MPIContext in IPython mode")
class TestLazyEval(MPIContextTestCase):

    ntargets = 'any'

    def test_single_add(self):
        a = self.context.zeros((53, 63))
        b = self.context.ones((53, 63))
        self.context.lazy = True
        c = a + b
        self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
        self.context.sync()
        self.context.lazy = False
        assert_array_equal(c.toarray(), a.toarray() + b.toarray())

    def test_single_add_with_context_manager(self):
        a = self.context.zeros((53, 63))
        b = self.context.ones((53, 63))
        with self.context.lazy_eval():
            c = a + b
            self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
        assert_array_equal(c.toarray(), a.toarray() + b.toarray())

    def test_single_mult(self):
        a = self.context.zeros((54, 64))
        b = self.context.ones((54, 64))
        with self.context.lazy_eval():
            c = a * b
            self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
        assert_array_equal(c.toarray(), a.toarray() * b.toarray())

    def test_constant_mult(self):
        a = self.context.zeros((55, 65))
        with self.context.lazy_eval():
            c = a * 2
            self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
        assert_array_equal(c.toarray(), a.toarray() * 2)

    def test_two_identical_add_expr(self):
        a = self.context.zeros((56, 66))
        b = self.context.ones((56, 66))
        with self.context.lazy_eval():
            c = a + b
            d = a + b
            self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
        assert_array_equal(c.toarray(), a.toarray() + b.toarray())
        assert_array_equal(d.toarray(), a.toarray() + b.toarray())

    def test_different_adds(self):
        a = self.context.zeros((58, 68))
        b = self.context.ones((58, 68))
        c = self.context.ones((58, 68)) + 1
        d = self.context.ones((58, 68)) + 2
        with self.context.lazy_eval():
            e = a + b
            f = c + d
            self.assertTrue(isinstance(e.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(f.key.dereference(), LazyPlaceholder))
        assert_array_equal(e.toarray(), a.toarray() + b.toarray())
        assert_array_equal(f.toarray(), c.toarray() + d.toarray())

    def test_more_different_adds(self):
        a = self.context.zeros((59, 69))
        b = self.context.ones((59, 69))
        c = self.context.ones((59, 69)) + 1
        with self.context.lazy_eval():
            e = a + b
            f = b + c
            self.assertTrue(isinstance(e.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(f.key.dereference(), LazyPlaceholder))
        assert_array_equal(e.toarray(), a.toarray() + b.toarray())
        assert_array_equal(f.toarray(), b.toarray() + c.toarray())

    def test_unary_ufuncs(self):
        a = self.context.ones((60, 70))
        b = -1 * self.context.ones((60, 70))
        with self.context.lazy_eval():
            c = -a
            d = gapi.absolute(b)
            self.assertTrue(isinstance(c.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
        assert_array_equal(c.toarray(), -a.toarray())
        assert_array_equal(d.toarray(), numpy.absolute(b.toarray()))

    def test_dependent_add(self):
        a = self.context.zeros((61, 71))
        b = self.context.ones((61, 71))
        c = self.context.ones((61, 71)) + 1
        with self.context.lazy_eval():
            t0 = a + b
            d = t0 + c
            self.assertTrue(isinstance(t0.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
        assert_array_equal(d.toarray(), a.toarray() + b.toarray() + c.toarray())

    def test_temporary_add(self):
        a = self.context.zeros((61, 71))
        b = self.context.ones((61, 71))
        c = self.context.ones((61, 71)) + 1
        with self.context.lazy_eval():
            d = a + b + c
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
        assert_array_equal(d.toarray(), a.toarray() + b.toarray() + c.toarray())

    def test_complex_expressions(self):
        a = self.context.zeros((52, 62))
        b = self.context.ones((52, 62))
        c = self.context.ones((52, 62)) + 1
        with self.context.lazy_eval():
            d = (2*a + (3*b + 4*c)) / 2
            e = gapi.negative(d * d)
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(e.key.dereference(), LazyPlaceholder))
        d_expected = (2*a.toarray() + (3*b.toarray() + 4*c.toarray())) / 2
        e_expected = numpy.negative(d * d)
        assert_array_equal(d.toarray(), d_expected)
        assert_array_equal(e.toarray(), e_expected)

    def test_lazy_creation(self):
        with self.context.lazy_eval():
            a = self.context.zeros((50, 60))
            b = self.context.zeros((50, 60))
        assert_array_equal(b.toarray(), numpy.zeros((50, 60)))
        assert_array_equal(a.toarray(), numpy.zeros((50, 60)))

    def test_creation_and_expressions(self):
        with self.context.lazy_eval():
            a = self.context.zeros((52, 62))
            b = self.context.ones((52, 62))
            c = self.context.ones((52, 62)) + 1
            d = (2*a + (3*b + 4*c)) / 2
            e = gapi.negative(d * d)
            self.assertTrue(isinstance(d.key.dereference(), LazyPlaceholder))
            self.assertTrue(isinstance(e.key.dereference(), LazyPlaceholder))
        d_expected = (2*a.toarray() + (3*b.toarray() + 4*c.toarray())) / 2
        e_expected = numpy.negative(d * d)
        assert_array_equal(d.toarray(), d_expected)
        assert_array_equal(e.toarray(), e_expected)

    def test_user_function(self):
        with self.context.lazy_eval():
            def local_square(la):
                return la * la
            da = self.context.ones((30, 40)) * 2
            new_key = self.context.apply(local_square, (da.key,), autoproxyize=True)[0]
            new_da = DistArray.from_localarrays(key=new_key,
                                                distribution=da.distribution,
                                                dtype=int)
        assert_array_equal(new_da.toarray(), (numpy.ones((30, 40)) * 2) ** 2)

    def test_multiple_return_values(self):
        with self.context.lazy_eval():
            da = self.context.ones((30, 40)) * 2
            def local_powers(la):
                return proxyize(la * la), proxyize(la * la * la)
            key0, key1 = self.context.apply(local_powers, (da.key,), nresults=2)[0]
            da0 = DistArray.from_localarrays(key=key0,
                                            distribution=da.distribution,
                                            dtype=int)
            self.assertTrue(isinstance(da0.key.dereference(), LazyPlaceholder))
            da1 = DistArray.from_localarrays(key=key1,
                                             distribution=da.distribution,
                                             dtype=int)
            self.assertTrue(isinstance(da1.key.dereference(), LazyPlaceholder))
        assert_array_equal(da0.toarray(), (numpy.ones((30, 40)) * 2) ** 2)
        assert_array_equal(da1.toarray(), (numpy.ones((30, 40)) * 2) ** 3)

    def test_lazy_loop(self):
        rvalues = []
        with self.context.lazy_eval():
            for i in range(10):
                rvalues.append(i * self.context.ones((10, 3)))

        for i, value in enumerate(rvalues):
            assert_array_equal(value.toarray(), i * numpy.ones((10, 3)))


class TestRegister(DefaultContextTestCase):

    ntargets = 'any'

    def test_local_add50(self):

        def local_add50(da):
            return da + 50
        self.context.register(local_add50)

        dc = self.context.local_add50(self.da)
        assert_allclose(dc.tondarray(), 2 * numpy.pi + 50)

    def test_local_add_num(self):

        def local_add_num(da, num):
            return da + num
        self.context.register(local_add_num)

        de = self.context.local_add_num(self.da, 11)
        assert_allclose(de.tondarray(), 2 * numpy.pi + 11)

    def test_local_sin(self):

        def local_sin(da):
            return numpy.sin(da)
        self.context.register(local_sin)

        db = self.context.local_sin(self.da)
        assert_allclose(0, db.tondarray(), atol=1e-14)

    def test_local_sum(self):

        def local_sum(da):
            return numpy.sum(da.ndarray)
        self.context.register(local_sum)

        dd = self.context.local_sum(self.da)
        if self.ntargets == 1:
            dd = [dd]
        lshapes = self.da.localshapes()
        expected = []
        for lshape in lshapes:
            expected.append(lshape[0] * lshape[1] * (2 * numpy.pi))
        for (v, e) in zip(dd, expected):
            self.assertAlmostEqual(v, e, places=5)

    def test_local_add_nums(self):

        def local_add_nums(da, num1, num2, num3):
            return da + num1 + num2 + num3
        self.context.register(local_add_nums)

        df = self.context.local_add_nums(self.da, 11, 12, 13)
        assert_allclose(df.tondarray(), 2 * numpy.pi + 11 + 12 + 13)

    def test_barrier(self):

        def call_barrier(da):
            da.comm.Barrier()
            return da
        self.context.register(call_barrier)

        self.context.call_barrier(self.da)

    def test_local_add_distarrayproxies(self):

        def local_add_distarrayproxies(da, dg):
            return da + dg
        self.context.register(local_add_distarrayproxies)

        dg = self.context.empty(self.da.distribution)
        dg.fill(33)
        dh = self.context.local_add_distarrayproxies(self.da, dg)
        assert_allclose(dh.tondarray(), 33 + 2 * numpy.pi)

    def test_local_add_mixed(self):

        def local_add_mixed(da, num1, dg, num2):
            return da + num1 + dg + num2
        self.context.register(local_add_mixed)

        di = self.context.empty(self.da.distribution)
        di.fill(33)
        dj = self.context.local_add_mixed(self.da, 11, di, 12)
        assert_allclose(dj.tondarray(), 2 * numpy.pi + 11 + 33 + 12)

    def test_local_add_kwargs(self):

        def local_add_kwargs(da, num1, num2=55):
            return da + num1 + num2
        self.context.register(local_add_kwargs)

        dl = self.context.local_add_kwargs(self.da, 11, num2=12)
        assert_allclose(dl.tondarray(), 2 * numpy.pi + 11 + 12)

    def test_local_add_supermix(self):

        def local_add_supermix(da, num1, db, num2, dc, num3=99, num4=66):
            return da + num1 + db + num2 + dc + num3 + num4
        self.context.register(local_add_supermix)

        dm = self.context.empty(self.da.distribution)
        dm.fill(22)
        dn = self.context.empty(self.da.distribution)
        dn.fill(44)
        do = self.context.local_add_supermix(self.da, 11, dm, 33, dc=dn, num3=55)
        expected = 2 * numpy.pi + 11 + 22 + 33 + 44 + 55 + 66
        assert_allclose(do.tondarray(), expected)

    def test_local_none(self):

        def local_none(da):
            return None
        self.context.register(local_none)

        dp = self.context.local_none(self.da)
        self.assertTrue(dp is None)

    def test_parameterless(self):

        def parameterless():
            """This is a parameterless function."""
            return None
        self.context.register(parameterless)

        self.assertRaises(TypeError, self.context.parameterless)

    @classmethod
    def setUpClass(cls):
        super(TestRegister, cls).setUpClass()
        cls.da = cls.context.empty((5, 5))
        cls.da.fill(2 * numpy.pi)

    def test_local(self):
        context = Context()

        shape = (4, 4)
        da = context.empty(shape)
        a = numpy.empty(shape)

        def fill_a(a):
            for (i, j), _ in numpy.ndenumerate(a):
                a[i, j] = i + j
            return a

        def fill_da(da):
            for i in da.distribution[0].global_iter:
                for j in da.distribution[1].global_iter:
                    da.global_index[i, j] = i + j
            return da

        self.context.register(fill_da)

        da = self.context.fill_da(da)
        a = fill_a(a)

        assert_array_equal(da.tondarray(), a)


class TestContext(DefaultContextTestCase):
    """Test Context methods"""

    @classmethod
    def setUpClass(cls):
        super(TestContext, cls).setUpClass()
        cls.ndarr = numpy.arange(16).reshape(4, 4)
        cls.darr = cls.context.fromndarray(cls.ndarr)

    def test_get_localarrays(self):
        las = self.darr.get_localarrays()
        self.assertIsInstance(las[0], LocalArray)

    def test_get_ndarrays(self):
        ndarrs = self.darr.get_ndarrays()
        self.assertIsInstance(ndarrs[0], numpy.ndarray)


@unittest.skipIf(not is_solo_mpi_process(),  # not in ipython mode
                 "Cannot test IPythonContext in MPI mode")
class TestIPythonContextCreation(IPythonContextTestCase):
    """Test Context Creation"""

    def test_create_Context(self):
        """Can we create a plain vanilla context?"""
        dac = Context(kind='IPython', client=self.client)
        self.assertIs(dac.client, self.client)

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        check_targets(required=2, available=len(self.client))
        dac = Context(self.client, targets=[0, 1])
        self.assertIs(dac.client, self.client)
        dac.close()

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        check_targets(required=4, available=len(self.client))
        targets = [3, 2]
        dac = Context(self.client, targets=targets)
        self.assertEqual(set(dac.targets), set(targets))
        dac.close()

    def test_context_target_reordering(self):
        """Are contexts' targets reordered in a consistent way?"""
        orig_targets = self.client.ids
        targets1 = orig_targets[:]
        targets2 = orig_targets[:]
        shuffle(targets1)
        shuffle(targets2)
        ctx1 = Context(self.client, targets=targets1)
        ctx2 = Context(self.client, targets=targets2)
        self.assertEqual(ctx1.targets, ctx2.targets)
        ctx1.close()
        ctx2.close()

    def test_create_delete_key(self):
        """ Check that a key can be created and then destroyed. """
        dac = Context(self.client)
        # Create and push a key/value.
        key, value = dac._generate_key(), 'test'
        dac._push({key: value}, targets=dac.targets)
        # Delete the key.
        dac.delete_key(key)
        dac.close()


@unittest.skipIf(is_solo_mpi_process(),  # not in MPI mode
                 "Cannot test MPIContext in IPython mode")
class TestMPIContextCreation(unittest.TestCase):
    """Test Context Creation"""

    def test_create_context(self):
        Context()

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        from distarray.globalapi.context import MPIContext
        check_targets(required=2, available=MPIContext.INTERCOMM.remote_size)
        Context(targets=[0, 1])

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        from distarray.globalapi.context import MPIContext
        check_targets(required=4, available=MPIContext.INTERCOMM.remote_size)
        targets = [3, 2]
        dac = Context(targets=targets)
        self.assertEqual(set(dac.targets), set(targets))

    def test_create_delete_key(self):
        """ Check that a key can be created and then destroyed. """
        dac = Context()
        # Create and push a key/value.
        key, value = dac._generate_key(), 'test'
        dac._push({key: value}, targets=dac.targets)
        # Delete the key.
        dac.delete_key(key)
        dac.close()


class TestPrimeCluster(DefaultContextTestCase):

    ntargets = 3

    def test_1D(self):
        a = self.context.empty((3,))
        self.assertEqual(a.grid_shape, (3,))

    def test_2D(self):
        a = self.context.empty((3, 3))
        db = Distribution(self.context, (3, 3), dist=('n', 'b'))
        b = self.context.empty(db)
        self.assertEqual(a.grid_shape, (3, 1))
        self.assertEqual(b.grid_shape, (1, 3))

    def test_3D(self):
        a = self.context.empty((3, 3, 3))
        db = Distribution(self.context, (3, 3, 3),
                                     dist=('n', 'b', 'n'))
        b = self.context.empty(db)
        dc = Distribution(self.context, (3, 3, 3),
                                     dist=('n', 'n', 'b'))
        c = self.context.empty(dc)
        self.assertEqual(a.grid_shape, (3, 1, 1))
        self.assertEqual(b.grid_shape, (1, 3, 1))
        self.assertEqual(c.grid_shape, (1, 1, 3))


class TestApply(DefaultContextTestCase):

    ntargets = 'any'

    def test_apply_no_args(self):

        def foo():
            return 42

        val = self.context.apply(foo)

        self.assertEqual(val, [42] * self.ntargets)

    def test_apply_pos_args(self):

        def foo(a, b, c):
            return a + b + c

        # push all arguments
        val = self.context.apply(foo, (1, 2, 3))
        self.assertEqual(val, [6] * self.ntargets)

        # some local, some pushed
        local_thing = self.context._key_and_push(2)[0]
        val = self.context.apply(foo, (1, local_thing, 3))

        self.assertEqual(val, [6] * self.ntargets)

        # all pushed
        local_args = self.context._key_and_push(1, 2, 3)
        val = self.context.apply(foo, local_args)

        self.assertEqual(val, [6] * self.ntargets)

    def test_apply_kwargs(self):

        def foo(a, b, c=None, d=None):
            c = -1 if c is None else c
            d = -2 if d is None else d
            return a + b + c + d

        # empty kwargs
        val = self.context.apply(foo, (1, 2))

        self.assertEqual(val, [0] * self.ntargets)

        # some empty
        val = self.context.apply(foo, (1, 2), {'d': 3})

        self.assertEqual(val, [5] * self.ntargets)

        # all kwargs
        val = self.context.apply(foo, (1, 2), {'c': 2, 'd': 3})

        self.assertEqual(val, [8] * self.ntargets)

        # now with local values
        local_a = self.context._key_and_push(1)[0]
        local_c = self.context._key_and_push(3)[0]

        val = self.context.apply(foo, (local_a, 2), {'c': local_c, 'd': 3})

        self.assertEqual(val, [9] * self.ntargets)

    def test_apply_proxy(self):

        def foo():
            return proxyize(10)  # noqa
        name = self.context.apply(foo)[0]

        def bar(obj):
            return obj + 10
        val = self.context.apply(bar, (name,))

        self.assertEqual(val, [20] * self.ntargets)

    def test_apply_proxyize_sync(self):

        def foo():
            p1 = proxyize(10)  # noqa
            p2 = proxyize(20)  # noqa
            return p1, 6, p2
        res = self.context.apply(foo)
        self.assertEqual(set(r[0].name for r in res), set([res[0][0].name]))
        self.assertEqual(set(r[-1].name for r in res), set([res[0][-1].name]))

    def test_apply_distarray(self):

        da = self.context.empty((len(self.context.targets),), dtype=numpy.uint32)

        def local_label(la):
            la.ndarray.fill(la.comm.rank)

        # Testing that we can pass in `da` and `apply()` extracts `da.key` automatically.
        self.context.apply(local_label, (da,))
        assert_array_equal(da.tondarray(), range(len(self.context.targets)))

        self.context.apply(local_label, kwargs={'la': da})
        assert_array_equal(da.tondarray(), range(len(self.context.targets)))


class TestGetBaseComm(DefaultContextTestCase):

    ntargets = 'any'

    def test_get_base_comm(self):

        def local_test_type():
            from mpi4py.MPI import Intracomm
            from distarray.localapi.mpiutils import get_base_comm
            bc = get_base_comm()
            return isinstance(bc, Intracomm), bc.rank, bc.size

        test, ranks, sizes = zip(*self.context.apply(local_test_type, ()))

        self.assertTrue(all(test))
        self.assertSetEqual(set(ranks), set(range(len(ranks))))
        self.assertSetEqual(set(sizes), set([len(sizes)]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
