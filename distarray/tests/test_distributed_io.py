# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Tests for distributed IO.

Many of these tests require a 4-engine cluster to be running locally, and will
write out temporary files.
"""

import unittest
import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from distarray.externals.six.moves import range

from distarray.client import DistArray
from distarray.context import Context
from distarray.testing import import_or_skip, temp_filepath, IpclusterTestCase


class TestFlatFileIO(IpclusterTestCase):

    def test_save_load_with_filenames(self):
        dac = Context(self.client)
        da = dac.empty((100,), dist={0: 'b'})

        output_paths = [temp_filepath() for target in dac.targets]
        try:
            dac.save(output_paths, da)
            db = dac.load(output_paths)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(da, db)
        finally:
            for filepath in output_paths:
                if os.path.exists(filepath):
                    os.remove(filepath)

    def test_save_load_with_prefix(self):
        dac = Context(self.client)
        da = dac.empty((100,), dist={0: 'b'})

        output_path = temp_filepath()
        try:
            dac.save(output_path, da)
            db = dac.load(output_path)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(da, db)
        finally:
            for rank in dac.targets:
                filepath = output_path + "_" + str(rank) + ".dnpy"
                if os.path.exists(filepath):
                    os.remove(filepath)


class TestHDF5FileIO(IpclusterTestCase):

    def test_write_block(self):
        h5py = import_or_skip('h5py')
        datalen = 33
        dac = Context(self.client)
        da = dac.empty((datalen,), dist={0: 'b'})
        for i in range(datalen):
            da[i] = i

        output_path = temp_filepath('.hdf5')

        try:
            dac.save_hdf5(output_path, da, mode='w')

            self.assertTrue(os.path.exists(output_path))

            with h5py.File(output_path, 'r') as fp:
                self.assertTrue("buffer" in fp)
                expected = np.arange(datalen)
                assert_equal(expected, fp["buffer"])

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_write_3d(self):
        h5py = import_or_skip('h5py')
        shape = (4, 5, 3)
        source = np.random.random(shape)

        dac = Context(self.client)
        dist = {0: 'b', 1: 'c', 2: 'n'}
        da = dac.empty(shape, dist=dist)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    da[i, j, k] = source[i, j, k]

        output_path = temp_filepath('.hdf5')

        try:
            dac.save_hdf5(output_path, da, mode='w')

            self.assertTrue(os.path.exists(output_path))

            with h5py.File(output_path, 'r') as fp:
                self.assertTrue("buffer" in fp)
                assert_allclose(source, fp["buffer"])

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_writing_two_datasets(self):
        h5py = import_or_skip('h5py')

        datalen = 33
        dac = Context(self.client)
        da = dac.empty((datalen,), dist={0: 'b'})

        for i in range(datalen):
            da[i] = i

        output_path = temp_filepath('.hdf5')

        try:
            # make a file, and write to dataset 'foo'
            with h5py.File(output_path, 'w') as fp:
                fp['foo'] = np.arange(10)

            # try saving to a different dataset
            dac.save_hdf5(output_path, da, key='bar', mode='a')

            with h5py.File(output_path, 'r') as fp:
                self.assertTrue("foo" in fp)
                self.assertTrue("bar" in fp)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
