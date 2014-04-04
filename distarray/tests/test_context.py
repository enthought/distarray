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
from distarray.local import LocalArray
from distarray.creation import fromarray
from distarray.testing import IpclusterTestCase


class TestContext(unittest.TestCase):
    """Test Context methods"""

    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        cls.ndarr = numpy.arange(16).reshape(4, 4)
        cls.darr = fromarray(cls.ndarr)

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


class TestContextCreation(IpclusterTestCase):
    """Test Context Creation"""

    def test_create_Context(self):
        """Can we create a plain vanilla context?"""
        dac = Context(self.client)
        self.assertIs(dac.client, self.client)
        del dac

    def test_create_Context_with_targets(self):
        """Can we create a context with a subset of engines?"""
        dac = Context(self.client, targets=[0, 1])
        self.assertIs(dac.client, self.client)

    def test_create_Context_with_targets_ranks(self):
        """Check that the target <=> rank mapping is consistent."""
        targets = [3, 2]
        dac = Context(self.client, targets=targets)
        self.assertEqual(set(dac.targets), set(targets))

    def test_context_target_reordering(self):
        '''Are contexts' targets reordered in a consistent way?'''
        orig_targets = self.client.ids
        ctx1 = Context(self.client, targets=shuffle(orig_targets[:]))
        ctx2 = Context(self.client, targets=shuffle(orig_targets[:]))
        self.assertEqual(ctx1.targets, ctx2.targets)

    def test_create_delete_key(self):
        """ Check that a key can be created and then destroyed. """
        dac = Context(self.client)
        # Create and push a key/value.
        key, value = dac._generate_key(), 'test'
        dac._push({key: value})
        # Delete the key.
        dac.delete_key(key)

    def test_purge_and_dump_keys(self):
        """ Check that we can get the existing keys and purge them. """
        # Get initial key count (probably 0).
        context0 = Context(self.client)
        num_keys0 = len(context0.dump_keys(all_other_contexts=True))
        # Create a context get the count on the new context.
        dac = Context(self.client)
        num_keys1 = len(dac.dump_keys())
        # Create and push a key.
        key = dac._generate_key()
        dac._execute('%s = 42' % (key))
        # Size of list of keys should have grown, both from the
        # all other context, and just the one context, point of view.
        num_keys2 = len(context0.dump_keys(all_other_contexts=True))
        self.assertGreater(num_keys2, num_keys0)
        num_keys3 = len(dac.dump_keys())
        self.assertGreater(num_keys3, num_keys1)
        # Delete the context.
        del dac
        # Key count should return to start.
        num_keys2 = len(context0.dump_keys(all_other_contexts=True))
        self.assertEqual(num_keys2, num_keys0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
