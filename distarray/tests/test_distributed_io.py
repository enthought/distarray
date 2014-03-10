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


class TestDnpyFileIO(IpclusterTestCase):

    def test_save_load_with_filenames(self):
        dac = Context(self.dv)
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
        dac = Context(self.dv)
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


bn_test_data = [
        ({'size': 2,
          'dist_type': 'b',
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
          'start': 0,
          'stop': 1,
         },
         {'size': 10,
          'dist_type': 'n',
         }),
        ({'size': 2,
          'dist_type': 'b',
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
          'start': 1,
          'stop': 2,
         },
         {'size': 10,
          'dist_type': 'n',
         })
     ]

nc_test_data = [
        ({'size': 10,
          'dist_type': 'n',
         },
         {'size': 2,
          'dist_type': 'c',
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
          'start': 0,
         },),

        ({'size': 10,
          'dist_type': 'n',
         },
         {'size': 2,
          'dist_type': 'c',
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
          'start': 1,
         },)
     ]

u_test_data = [
        # Note: indices must be in increasing order
        #       (restiction of h5py / HDF5)

        ({'size': 20,
          'dist_type': 'u',
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
          'indices': [0, 3, 4, 6, 8, 10, 13, 15, 18],
         },),
        ({'size': 20,
          'dist_type': 'u',
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
          'indices': [1, 2, 5, 7, 9, 11, 12, 14, 16, 17, 19],
         },)
    ]


class TestNpyFileIO(IpclusterTestCase):

    @classmethod
    def get_ipcluster_size(cls):
        return 2

    def test_load_bn(self):

        # set up test file
        output_path = temp_filepath('.npy')
        expected = np.arange(20).reshape(2, 10)
        np.save(output_path, expected)

        # load it in with load_npy
        dac = Context(self.dv, targets=[0, 1])
        dim_datas = bn_test_data

        try:
            da = dac.load_npy(output_path, dim_datas)
            assert_equal(da, expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_load_nc(self):

        # set up test file
        output_path = temp_filepath('.npy')
        expected = np.arange(20).reshape(2, 10)
        np.save(output_path, expected)

        # load it in with load_npy
        dac = Context(self.dv, targets=[0, 1])
        dim_datas = nc_test_data

        try:
            da = dac.load_npy(output_path, dim_datas)
            assert_equal(da, expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_load_u(self):

        # set up test file
        output_path = temp_filepath('.npy')
        expected = np.arange(20)
        np.save(output_path, expected)

        # load it in with load_npy
        dac = Context(self.dv, targets=[0, 1])

        dim_datas = u_test_data

        try:
            da = dac.load_npy(output_path, dim_datas)
            assert_equal([x for x in da], expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestHDF5FileIO(IpclusterTestCase):

    def test_save_block(self):
        h5py = import_or_skip('h5py')
        datalen = 33
        dac = Context(self.dv)
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

    def test_save_3d(self):
        h5py = import_or_skip('h5py')
        shape = (4, 5, 3)
        source = np.random.random(shape)

        dac = Context(self.dv)
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

    def test_save_two_datasets(self):
        h5py = import_or_skip('h5py')

        datalen = 33
        dac = Context(self.dv)
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

    def test_load_bn(self):
        h5py = import_or_skip('h5py')

        # set up test file
        output_path = temp_filepath()
        expected = np.arange(20).reshape(2, 10)
        with h5py.File(output_path, 'w') as fp:
            fp["load_test"] = expected

        # load it in with load_hdf5
        dac = Context(self.dv, targets=[0, 1])

        dim_datas = bn_test_data

        try:
            da = dac.load_hdf5(output_path, dim_datas, key="load_test")
            assert_equal(da, expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_load_nc(self):
        h5py = import_or_skip('h5py')

        # set up test file
        output_path = temp_filepath()
        expected = np.arange(20).reshape(2, 10)
        with h5py.File(output_path, 'w') as fp:
            fp["load_test"] = expected

        # load it in with load_hdf5
        dac = Context(self.dv, targets=[0, 1])

        dim_datas = nc_test_data

        try:
            da = dac.load_hdf5(output_path, dim_datas, key="load_test")
            assert_equal(da, expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_load_u(self):
        h5py = import_or_skip('h5py')

        # set up test file
        output_path = temp_filepath()
        expected = np.arange(20)
        with h5py.File(output_path, 'w') as fp:
            fp["load_test"] = expected

        # load it in with load_hdf5
        dac = Context(self.dv, targets=[0, 1])

        dim_datas = u_test_data

        try:
            da = dac.load_hdf5(output_path, dim_datas, key="load_test")
            assert_equal([x for x in da], expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
