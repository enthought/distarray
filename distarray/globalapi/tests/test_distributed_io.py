# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Tests for distributed IO.

Many of these tests require a 4-engine cluster running and will write out (and
afterwards remove) temporary files.  These tests assume that all engines have
access to the same filesystem but do not assume the client has access to that
same filesystem.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from distarray.externals.six.moves import range

from distarray.testing import import_or_skip, DefaultContextTestCase
from distarray.globalapi.distarray import DistArray
from distarray.globalapi.maps import Distribution


def cleanup_file(filepath):
    import os
    if os.path.exists(filepath):
        os.remove(filepath)


def engine_temp_path(extension=''):
    from distarray.testing import temp_filepath
    return temp_filepath(extension)


class TestDnpyFileIO(DefaultContextTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDnpyFileIO, cls).setUpClass()
        cls.distribution = Distribution(cls.context, (100,), dist={0: 'b'})
        cls.da = cls.context.empty(cls.distribution)
        cls.output_paths = cls.context.apply(engine_temp_path)

    def test_save_load_with_filenames(self):

        try:
            self.context.save_dnpy(self.output_paths, self.da)
            db = self.context.load_dnpy(self.output_paths)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(self.da, db)
        finally:
            for filepath, target in zip(self.output_paths, self.context.targets):
                self.context.apply(cleanup_file, (filepath,), targets=(target,))

    def test_save_load_with_prefix(self):

        output_path = self.output_paths[0]
        try:
            self.context.save_dnpy(output_path, self.da)
            db = self.context.load_dnpy(output_path)
            self.assertTrue(isinstance(db, DistArray))
            self.assertEqual(self.da, db)
        finally:
            for rank in self.context.targets:
                filepath = output_path + "_" + str(rank) + ".dnpy"
                self.context.apply(cleanup_file, (filepath,), targets=(rank,))


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
        #       (restriction of h5py / HDF5)

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


class TestNpyFileLoad(DefaultContextTestCase):

    """Try loading a .npy file on the engines.

    This test assumes that all engines have access to the same file system.
    """

    ntargets = 2

    @classmethod
    def setUpClass(cls):
        super(TestNpyFileLoad, cls).setUpClass()
        cls.expected = np.arange(20).reshape(2, 10)

        def save_test_file(data):
            import numpy
            from distarray.testing import temp_filepath
            output_path = temp_filepath('.npy')
            numpy.save(output_path, data)
            return output_path

        cls.output_path = cls.context.apply(save_test_file, (cls.expected,),
                                            targets=[cls.context.targets[0]])[0]  # noqa

    @classmethod
    def tearDownClass(cls):
        cls.context.apply(cleanup_file, (cls.output_path,),
                          targets=[cls.context.targets[0]])
        super(TestNpyFileLoad, cls).tearDownClass()

    def test_load_bn(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           bn_test_data)
        da = self.context.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nc(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           nc_test_data)
        da = self.context.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])

    def test_load_nu(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           nu_test_data)
        da = self.context.load_npy(self.output_path, distribution)
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                self.assertEqual(da[i, j], self.expected[i, j])


def check_hdf5_file(output_path, expected, dataset="buffer"):
    import h5py
    import numpy

    with h5py.File(output_path, 'r') as fp:
        if dataset not in fp:
            return False
        if not numpy.array_equal(expected, fp[dataset]):
            return False

    return True


class TestHdf5FileSave(DefaultContextTestCase):

    def setUp(self):
        super(TestHdf5FileSave, self).setUp()
        self.h5py = import_or_skip('h5py')
        self.output_path = self.context.apply(engine_temp_path, ('.hdf5',),
                                              targets=[self.context.targets[0]])[0]

    def tearDown(self):
        self.context.apply(cleanup_file, (self.output_path,),
                           targets=[self.context.targets[0]])

    def test_save_block(self):
        datalen = 33
        expected = np.arange(datalen)
        da = self.context.fromarray(expected)
        self.context.save_hdf5(self.output_path, da, mode='w')

        file_check = self.context.apply(check_hdf5_file,
                                        (self.output_path, expected),
                                        targets=[self.context.targets[0]])[0]
        self.assertTrue(file_check)

    def test_save_3d(self):
        shape = (4, 5, 3)
        expected = np.random.random(shape)

        dist = {0: 'b', 1: 'c', 2: 'n'}
        distribution = Distribution(self.context, shape, dist=dist)
        da = self.context.fromarray(expected, distribution=distribution)

        self.context.save_hdf5(self.output_path, da, mode='w')
        file_check = self.context.apply(check_hdf5_file,
                                        (self.output_path, expected),
                                        targets=[self.context.targets[0]])[0]
        self.assertTrue(file_check)

    def test_save_two_datasets(self):
        datalen = 33
        foo = np.arange(datalen)
        bar = np.random.random(datalen)
        da_foo = self.context.fromarray(foo)
        da_bar = self.context.fromarray(bar)

        # save 'foo' to a file
        self.context.save_hdf5(self.output_path, da_foo, key='foo', mode='w')

        # save 'bar' to a different dataset in the same file
        self.context.save_hdf5(self.output_path, da_bar, key='bar', mode='a')

        foo_checks = self.context.apply(check_hdf5_file,
                                        (self.output_path, foo),
                                        {'dataset': 'foo'},
                                        targets=[self.context.targets[0]])[0]
        self.assertTrue(foo_checks)
        bar_checks = self.context.apply(check_hdf5_file,
                                        (self.output_path, bar),
                                        {'dataset': 'bar'},
                                        targets=[self.context.targets[0]])[0]
        self.assertTrue(bar_checks)


class TestHdf5FileLoad(DefaultContextTestCase):

    ntargets = 2

    @classmethod
    def setUpClass(cls):
        cls.h5py = import_or_skip('h5py')
        super(TestHdf5FileLoad, cls).setUpClass()
        cls.output_path = cls.context.apply(engine_temp_path, ('.hdf5',),
                                            targets=[cls.context.targets[0]])[0]
        cls.expected = np.arange(20).reshape(2, 10)

        def make_test_file(output_path, arr):
            import h5py
            with h5py.File(output_path, 'w') as fp:
                fp["test"] = arr

        cls.context.apply(make_test_file, (cls.output_path, cls.expected),
                      targets=[cls.context.targets[0]])

    @classmethod
    def tearDownClass(cls):
        cls.context.apply(cleanup_file, (cls.output_path,),
                      targets=[cls.context.targets[0]])
        super(TestHdf5FileLoad, cls).tearDownClass()

    def test_load_bn(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           bn_test_data)
        da = self.context.load_hdf5(self.output_path, distribution, key="test")
        assert_array_equal(self.expected, da)

    def test_load_nc(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           nc_test_data)
        da = self.context.load_hdf5(self.output_path, distribution, key="test")
        assert_array_equal(self.expected, da)

    def test_load_nu(self):
        distribution = Distribution.from_dim_data_per_rank(self.context,
                                                           nu_test_data)
        da = self.context.load_hdf5(self.output_path, distribution, key="test")
        assert_array_equal(self.expected, da)


if __name__ == '__main__':
    unittest.main(verbosity=2)
