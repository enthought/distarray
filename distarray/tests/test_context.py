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

from distarray import Context
from distarray.ipython_utils import IPythonClient
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
        '''Are contexts' targets reordered in a consistent way?'''
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
        a = self.context.empty((3,))
        self.assertEqual(a.grid_shape, (3,))

    def test_2D(self):
        a = self.context.empty((3, 3))
        b = self.context.empty((3, 3), dist=('n', 'b'))
        self.assertEqual(a.grid_shape, (3, 1))
        self.assertEqual(b.grid_shape, (1, 3))

    def test_3D(self):
        a = self.context.empty((3, 3, 3))
        b = self.context.empty((3, 3, 3), dist=('n', 'b', 'n'))
        c = self.context.empty((3, 3, 3), dist=('n', 'n', 'b'))
        self.assertEqual(a.grid_shape, (3, 1, 1))
        self.assertEqual(b.grid_shape, (1, 3, 1))
        self.assertEqual(c.grid_shape, (1, 1, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
