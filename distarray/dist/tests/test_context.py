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

from distarray.dist.context import Context
from distarray.dist.maps import Distribution
from distarray.dist.ipython_utils import IPythonClient
from distarray.local import LocalArray


class TestContext(unittest.TestCase):
    """Test Context methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.ndarr = numpy.arange(16).reshape(4, 4)
        cls.darr = cls.context.fromndarray(cls.ndarr)

    @classmethod
    def tearDownClass(cls):
        """Close the client connections"""
        cls.context.close()

    def test_get_localarrays(self):
        las = self.darr.get_localarrays()
        self.assertIsInstance(las[0], LocalArray)

    def test_get_ndarrays(self):
        ndarrs = self.darr.get_ndarrays()
        self.assertIsInstance(ndarrs[0], numpy.ndarray)


class TestContextCreation(unittest.TestCase):
    """Test Context Creation"""

    def test_create_Context(self):
        """Can we create a plain vanilla context?"""
        client = IPythonClient()
        dac = Context(client)
        self.assertIs(dac.client, client)
        dac.close()
        client.close()

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        client = IPythonClient()
        dac = Context(client, targets=[0, 1])
        self.assertIs(dac.client, client)
        dac.close()
        client.close()

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        client = IPythonClient()
        targets = [3, 2]
        dac = Context(client, targets=targets)
        self.assertEqual(set(dac.targets), set(targets))
        dac.close()
        client.close()

    def test_context_target_reordering(self):
        """Are contexts' targets reordered in a consistent way?"""
        client = IPythonClient()
        orig_targets = client.ids
        ctx1 = Context(client, targets=shuffle(orig_targets[:]))
        ctx2 = Context(client, targets=shuffle(orig_targets[:]))
        self.assertEqual(ctx1.targets, ctx2.targets)
        ctx1.close()
        ctx2.close()
        client.close()

    def test_create_delete_key(self):
        """ Check that a key can be created and then destroyed. """
        client = IPythonClient()
        dac = Context(client)
        # Create and push a key/value.
        key, value = dac._generate_key(), 'test'
        dac._push({key: value})
        # Delete the key.
        dac.delete_key(key)
        dac.close()
        client.close()


class TestPrimeCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = Context(targets=range(3))

    @classmethod
    def tearDownClass(cls):
        cls.context.close()

    def test_1D(self):
        d = Distribution.from_shape(self.context, (3,))
        a = self.context.empty(d)
        self.assertEqual(a.grid_shape, (3,))

    def test_2D(self):
        da = Distribution.from_shape(self.context, (3, 3))
        a = self.context.empty(da)
        db = Distribution.from_shape(self.context, (3, 3), dist=('n', 'b'))
        b = self.context.empty(db)
        self.assertEqual(a.grid_shape, (3, 1))
        self.assertEqual(b.grid_shape, (1, 3))

    def test_3D(self):
        da = Distribution.from_shape(self.context, (3, 3, 3))
        a = self.context.empty(da)
        db = Distribution.from_shape(self.context, (3, 3, 3),
                                     dist=('n', 'b', 'n'))
        b = self.context.empty(db)
        dc = Distribution.from_shape(self.context, (3, 3, 3),
                                     dist=('n', 'n', 'b'))
        c = self.context.empty(dc)
        self.assertEqual(a.grid_shape, (3, 1, 1))
        self.assertEqual(b.grid_shape, (1, 3, 1))
        self.assertEqual(c.grid_shape, (1, 1, 3))


class TestApply(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = Context()

    def test_apply_no_args(self):

        def foo():
            return 42

        val = self.context.apply(foo)

        self.assertEqual(val, [42]*4)

    def test_apply_pos_args(self):

        def foo(a, b, c):
            return a + b + c

        # push all arguments
        val = self.context.apply(foo, (1, 2, 3))
        self.assertEqual(val, [6]*4)

        # some local, some pushed
        local_thing = self.context._key_and_push(2)[0]
        val = self.context.apply(foo, (1, local_thing, 3))

        self.assertEqual(val, [6]*4)

        # all pushed
        local_args = self.context._key_and_push(1, 2, 3)
        val = self.context.apply(foo, local_args)

        self.assertEqual(val, [6]*4)

    def test_apply_kwargs(self):

        def foo(a, b, c=None, d=None):
            c = -1 if c is None else c
            d = -2 if d is None else d
            return a + b + c + d

        # empty kwargs
        val = self.context.apply(foo, (1, 2))

        self.assertEqual(val, [0]*4)

        # some empty
        val = self.context.apply(foo, (1, 2), {'d': 3})

        self.assertEqual(val, [5]*4)

        # all kwargs
        val = self.context.apply(foo, (1, 2), {'c': 2, 'd': 3})

        self.assertEqual(val, [8]*4)

        # now with local values
        local_a = self.context._key_and_push(1)[0]
        local_c = self.context._key_and_push(3)[0]

        val = self.context.apply(foo, (local_a, 2), {'c': local_c, 'd': 3})

        self.assertEqual(val, [9]*4)

    def test_apply_return_val(self):

        def foo(a, b, c=None):
            c = 3 if c is None else c
            return a + b + c

        name = self.context.apply(foo, (1, 2), {'c': 5}, result_name='test')

        self.assertEqual(name, 'test')

        val = self.context._pull(name)

        self.assertEqual(val, [8]*len(self.context.targets))


if __name__ == '__main__':
    unittest.main(verbosity=2)
