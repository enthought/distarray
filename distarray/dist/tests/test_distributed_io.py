# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

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

from distarray.testing import import_or_skip, temp_filepath
from distarray.dist.distarray import DistArray
from distarray.dist.context import Context
from distarray.dist.maps import Distribution


def cleanup_file(filepath):
    import os
    if os.path.exists(filepath):
        os.remove(filepath)


class TestDnpyFileIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dac = Context()
        cls.distribution = Distribution.from_shape(cls.dac, (100,),
                                                   dist={0: 'b'})
        cls.da = cls.dac.empty(cls.distribution)

        def engine_temp_path():
            from distarray.testing import temp_filepath
            return temp_filepath()

        cls.output_paths = cls.dac.apply(engine_temp_path, return_proxy=False)

    def test_save_load_with_filenames(self):

        try:
            self.dac.save_dnpy(self.output_paths, self.da)
            db = self.dac.load_dnpy(self.output_paths)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(self.da, db)
        finally:
            for filepath, target in zip(self.output_paths, self.dac.targets):
                self.dac.apply(cleanup_file, (filepath,), targets=(target,))

    def test_save_load_with_prefix(self):

        output_path = self.output_paths[0]
        try:
            self.dac.save_dnpy(output_path, self.da)
            db = self.dac.load_dnpy(output_path)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(self.da, db)
        finally:
            for rank in self.dac.targets:
                filepath = output_path + "_" + str(rank) + ".dnpy"
                self.dac.apply(cleanup_file, (filepath,), targets=(rank,))


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


class TestNpyFileLoad(unittest.TestCase):

    """Try loading a .npy file on the engines.

    This test assumes that all engines have access to the same file system.
    """

    @classmethod
    def setUpClass(cls):
        cls.dac = Context(targets=[0, 1])
        cls.expected = np.arange(20).reshape(2, 10)

        def save_test_file(data):
            import numpy
            from distarray.testing import temp_filepath
            output_path = temp_filepath('.npy')
            numpy.save(output_path, data)
            return output_path

        cls.output_path = cls.dac.apply(save_test_file, (cls.expected,),
                                        return_proxy=False,
                                        targets=cls.dac.targets[0])

    @classmethod
    def tearDownClass(cls):
        cls.dac.apply(cleanup_file, (cls.output_path,),
                      targets=cls.dac.targets[0])

        # clean up the context keys
        cls.dac.close()

    def test_load_bn(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           bn_test_data)
        da = self.dac.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nc(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           nc_test_data)
        da = self.dac.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nu(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           nu_test_data)
        da = self.dac.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])


class TestHdf5FileSave(unittest.TestCase):

    def setUp(self):
        self.h5py = import_or_skip('h5py')
        self.dac = Context()

        def engine_temp_path():
            from distarray.testing import temp_filepath
            return temp_filepath('.hdf5')

        self.output_path = self.dac.apply(engine_temp_path,
                                          targets=self.dac.targets[0],
                                          return_proxy=False)

    def tearDown(self):
        self.dac.apply(cleanup_file, (self.output_path,),
                       targets=self.dac.targets[0])
        self.dac.close()

    def test_save_block(self):
        datalen = 33
        expected = np.arange(datalen)
        da = self.dac.fromarray(expected)
        self.dac.save_hdf5(self.output_path, da, mode='w')

        def check_file(output_path, expected):
            import h5py
            import numpy

            with h5py.File(output_path, 'r') as fp:
                if "buffer" not in fp:
                    return False
                if not numpy.array_equal(expected, fp["buffer"]):
                    return False

            return True

        file_check = self.dac.apply(check_file, (self.output_path, expected),
                                    targets=self.dac.targets[0],
                                    return_proxy=False)
        self.assertTrue(file_check)

    def test_save_3d(self):
        shape = (4, 5, 3)
        source = np.random.random(shape)

        dist = {0: 'b', 1: 'c', 2: 'n'}
        distribution = Distribution.from_shape(self.dac, shape, dist=dist)
        da = self.dac.empty(distribution)

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
        distribution = Distribution.from_shape(self.dac, (datalen,),
                                               dist={0: 'b'})
        da = self.dac.empty(distribution)

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


class TestHdf5FileLoad(unittest.TestCase):

    def setUp(self):
        self.h5py = import_or_skip('h5py')
        self.dac = Context(targets=[0, 1])
        self.output_path = temp_filepath('.hdf5')
        self.expected = np.arange(20).reshape(2, 10)
        with self.h5py.File(self.output_path, 'w') as fp:
            fp["test"] = self.expected

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.dac.close()

    def test_load_bn(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           bn_test_data)
        da = self.dac.load_hdf5(self.output_path, distribution, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])

    def test_load_nc(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           nc_test_data)
        da = self.dac.load_hdf5(self.output_path, distribution, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])

    def test_load_nu(self):
        distribution = Distribution.from_dim_data_per_rank(self.dac,
                                                           nu_test_data)
        da = self.dac.load_hdf5(self.output_path, distribution, key="test")
        for i, v in np.ndenumerate(self.expected):
            self.assertEqual(v, da[i])


if __name__ == '__main__':
    unittest.main(verbosity=2)
