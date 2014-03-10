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
        ({'size': 2,
          'dist_type': 'n',
         },
         {'size': 10,
          'dist_type': 'c',
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
          'start': 0,
         },),

        ({'size': 2,
          'dist_type': 'n',
         },
         {'size': 10,
          'dist_type': 'c',
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
          'start': 1,
         },)
     ]

nu_test_data = [
        # Note: unstructured indices must be in increasing order
        #       (restiction of h5py / HDF5)

        (
         {'size': 2,
          'dist_type': 'n',
         },
         {'size': 10,
          'dist_type': 'u',
          'proc_grid_rank': 0,
          'proc_grid_size': 2,
          'indices': [0, 3, 4, 6, 8],
         },
        ),
        (
         {'size': 2,
          'dist_type': 'n',
         },
         {'size': 10,
          'dist_type': 'u',
          'proc_grid_rank': 1,
          'proc_grid_size': 2,
          'indices': [1, 2, 5, 7, 9],
         },
        )
    ]


class TestNpyFileLoad(IpclusterTestCase):

    @classmethod
    def get_ipcluster_size(cls):
        return 2

    def setUp(self):
        self.dac = Context(self.client, targets=[0, 1])

        # make a test file
        self.output_path = temp_filepath('.npy')
        self.expected = np.arange(20).reshape(2, 10)
        np.save(self.output_path, self.expected)

    def tearDown(self):
        # delete the test file
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        super(TestNpyFileLoad, self).tearDown()

    def test_load_bn(self):
        dim_datas = bn_test_data
        da = self.dac.load_npy(self.output_path, dim_datas)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nc(self):
        dim_datas = nc_test_data
        da = self.dac.load_npy(self.output_path, dim_datas)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nu(self):
        dim_datas = nu_test_data
        da = self.dac.load_npy(self.output_path, dim_datas)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])


class TestHdf5FileSave(IpclusterTestCase):

    def setUp(self):
        self.h5py = import_or_skip('h5py')
        self.output_path = temp_filepath('.hdf5')
        self.dac = Context(self.client)

    def tearDown(self): 
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        super(TestHdf5FileSave, self).tearDown()

    def test_save_block(self):
        datalen = 33
        da = self.dac.empty((datalen,), dist={0: 'b'})
        for i in range(datalen):
            da[i] = i

        self.dac.save_hdf5(self.output_path, da, mode='w')
        with self.h5py.File(self.output_path, 'r') as fp:
            self.assertTrue("buffer" in fp)
            expected = np.arange(datalen)
            assert_equal(expected, fp["buffer"])

    def test_save_3d(self):
        shape = (4, 5, 3)
        source = np.random.random(shape)

        dist = {0: 'b', 1: 'c', 2: 'n'}
        da = self.dac.empty(shape, dist=dist)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    da[i, j, k] = source[i, j, k]

        self.dac.save_hdf5(self.output_path, da, mode='w')
        with self.h5py.File(self.output_path, 'r') as fp:
            self.assertTrue("buffer" in fp)
            assert_allclose(source, fp["buffer"])

    def test_save_two_datasets(self):
        datalen = 33
        da = self.dac.empty((datalen,), dist={0: 'b'})

        for i in range(datalen):
            da[i] = i

        # make a file, and write to dataset 'foo'
        with self.h5py.File(self.output_path, 'w') as fp:
            fp['foo'] = np.arange(10)

        # try saving to a different dataset
        self.dac.save_hdf5(self.output_path, da, key='bar', mode='a')

        with self.h5py.File(self.output_path, 'r') as fp:
            self.assertTrue("foo" in fp)
            self.assertTrue("bar" in fp)


class TestHdf5FileLoad(IpclusterTestCase):

    @classmethod
    def get_ipcluster_size(cls):
        return 2

    def setUp(self):
        self.h5py = import_or_skip('h5py')
        self.dac = Context(self.client, targets=[0, 1])
        self.output_path = temp_filepath('.hdf5')
        self.expected = np.arange(20).reshape(2, 10)
        with self.h5py.File(self.output_path, 'w') as fp:
            fp["test"] = self.expected

    def tearDown(self): 
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        super(TestHdf5FileLoad, self).tearDown()

    def test_load_bn(self):
        da = self.dac.load_hdf5(self.output_path, bn_test_data, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])

    def test_load_nc(self):
        da = self.dac.load_hdf5(self.output_path, nc_test_data, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])

    def test_load_nu(self):
        da = self.dac.load_hdf5(self.output_path, nu_test_data, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])


if __name__ == '__main__':
    unittest.main(verbosity=2)
