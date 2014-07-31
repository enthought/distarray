# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import os
import numpy

from numpy.testing import assert_allclose, assert_equal
from distarray.testing import ParallelTestCase, import_or_skip, temp_filepath
from distarray.localapi import LocalArray, ndenumerate
from distarray.localapi import (save_dnpy, load_dnpy, save_hdf5, load_hdf5,
                             load_npy)
from distarray.localapi.maps import Distribution


class TestDnpyFileIO(ParallelTestCase):

    def setUp(self):
        d = Distribution.from_shape(comm=self.comm, shape=(7,))
        self.larr0 = LocalArray(d)

        # a different file on every engine
        self.output_path = temp_filepath(extension='.dnpy')

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_flat_file_save_with_filename(self):
        save_dnpy(self.output_path, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    def test_flat_file_save_with_file_object(self):
        with open(self.output_path, 'wb') as fp:
            save_dnpy(fp, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    def test_flat_file_save_load_with_filename(self):
        save_dnpy(self.output_path, self.larr0)
        larr1 = load_dnpy(comm=self.comm, file=self.output_path)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)

    def test_flat_file_save_load_with_file_object(self):
        save_dnpy(self.output_path, self.larr0)
        with open(self.output_path, 'rb') as fp:
            larr1 = load_dnpy(comm=self.comm, file=fp)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)


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


class TestNpyFileLoad(ParallelTestCase):

    comm_size = 2

    def setUp(self):
        self.rank = self.comm.Get_rank()

        # set up a common filename to work with
        if self.rank == 0:
            self.output_path = temp_filepath(extension='.npy')
        else:
            self.output_path = None
        self.output_path = self.comm.bcast(self.output_path, root=0)

        # save some data to that file
        self.expected = numpy.arange(20).reshape(2, 10)

        if self.rank == 0:
            numpy.save(self.output_path, self.expected)
        self.comm.Barrier()

    def tearDown(self):
        # delete the test file
        if self.rank == 0:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def test_load_bn(self):
        dim_data_per_rank = bn_test_data
        la = load_npy(comm=self.comm, filename=self.output_path,
                      dim_data=dim_data_per_rank[self.rank])
        assert_equal(la, self.expected[numpy.newaxis, self.rank])

    def test_load_nc(self):
        dim_data_per_rank = nc_test_data
        expected_slices = [(slice(None), slice(0, None, 2)),
                           (slice(None), slice(1, None, 2))]

        la = load_npy(comm=self.comm, filename=self.output_path,
                      dim_data=dim_data_per_rank[self.rank])
        assert_equal(la, self.expected[expected_slices[self.rank]])

    def test_load_nu(self):
        dim_data_per_rank = nu_test_data
        expected_indices = [dd[1]['indices'] for dd in dim_data_per_rank]

        la = load_npy(comm=self.comm, filename=self.output_path,
                      dim_data=dim_data_per_rank[self.rank])
        assert_equal(la, self.expected[:, expected_indices[self.rank]])


class TestHdf5FileSave(ParallelTestCase):

    def setUp(self):
        self.rank = self.comm.Get_rank()
        self.h5py = import_or_skip('h5py')
        self.key = "data"

        # set up a common file to work with
        if self.rank == 0:
            self.output_path = temp_filepath(extension='.hdf5')
        else:
            self.output_path = None
        self.output_path = self.comm.bcast(self.output_path, root=0)

    def test_save_1d(self):
        d = Distribution.from_shape(comm=self.comm, shape=(51,))
        la = LocalArray(d)
        np_arr = numpy.random.random(la.local_shape)
        la.ndarray = np_arr
        save_hdf5(self.output_path, la, key=self.key, mode='w')

        # check saved file
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            for i, v in ndenumerate(la):
                self.assertEqual(v, fp[self.key][i])

    def test_save_2d(self):
        d = Distribution.from_shape(comm=self.comm, shape=(11, 15))
        la = LocalArray(d)
        np_arr = numpy.random.random(la.local_shape)
        la.ndarray = np_arr
        save_hdf5(self.output_path, la, key=self.key, mode='w')
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            for i, v in ndenumerate(la):
                self.assertEqual(v, fp[self.key][i])

    def tearDown(self):
        # delete the test file
        if self.rank == 0:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)


class TestHdf5FileLoad(ParallelTestCase):

    comm_size = 2

    def setUp(self):
        self.rank = self.comm.Get_rank()
        self.h5py = import_or_skip('h5py')
        self.key = "data"
        self.expected = numpy.arange(20).reshape(2, 10)

        # set up a common file to work with
        if self.rank == 0:
            self.output_path = temp_filepath(extension='.hdf5')
            with self.h5py.File(self.output_path, 'w') as fp:
                fp[self.key] = self.expected
        else:
            self.output_path = None
        self.comm.Barrier() # wait until file exists
        self.output_path = self.comm.bcast(self.output_path, root=0)

    def tearDown(self):
        # delete the test file
        if self.rank == 0:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def test_load_bn(self):
        dim_data_per_rank = bn_test_data
        la = load_hdf5(comm=self.comm, filename=self.output_path,
                       dim_data=dim_data_per_rank[self.rank],
                       key=self.key)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            assert_equal(la, self.expected[numpy.newaxis, self.rank])

    def test_load_nc(self):
        dim_data_per_rank = nc_test_data
        expected_slices = [(slice(None), slice(0, None, 2)),
                           (slice(None), slice(1, None, 2))]
        la = load_hdf5(comm=self.comm, filename=self.output_path,
                       dim_data=dim_data_per_rank[self.rank],
                       key=self.key)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            expected_slice = expected_slices[self.rank]
            assert_equal(la, self.expected[expected_slice])

    def test_load_nu(self):
        dim_data_per_rank = nu_test_data
        expected_indices = [dd[1]['indices'] for dd in dim_data_per_rank]
        la = load_hdf5(comm=self.comm, filename=self.output_path,
                       dim_data=dim_data_per_rank[self.rank],
                       key=self.key)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            assert_equal(la, self.expected[:, expected_indices[self.rank]])
