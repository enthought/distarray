import tempfile
import os
import numpy

from numpy.testing import assert_allclose, assert_equal
from distarray.testing import MpiTestCase, import_or_skip, temp_filepath
from distarray.local import LocalArray, ndenumerate
from distarray.local import save, load, save_hdf5, load_hdf5, load_npy


class TestDnpyFileIO(MpiTestCase):

    def setUp(self):
        self.larr0 = LocalArray((7,), comm=self.comm)

        # a different file on every engine
        self.output_path = temp_filepath(extension='.dnpy')

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_flat_file_save_with_filename(self):
        save(self.output_path, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    def test_flat_file_save_with_file_object(self):
        with open(self.output_path, 'wb') as fp:
            save(fp, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    def test_flat_file_save_load_with_filename(self):
        save(self.output_path, self.larr0)
        larr1 = load(self.output_path, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)

    def test_flat_file_save_load_with_file_object(self):
        save(self.output_path, self.larr0)
        with open(self.output_path, 'rb') as fp:
            larr1 = load(fp, comm=self.comm)
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


class TestNpyFileLoad(MpiTestCase):

    @classmethod
    def get_comm_size(self):
        return 2

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
        dim_datas = bn_test_data
        la = load_npy(self.output_path, dim_datas[self.rank], comm=self.comm)
        assert_equal(la, self.expected[numpy.newaxis, self.rank])

    def test_load_nc(self):
        dim_datas = nc_test_data
        expected_slices = [(slice(None), slice(0, None, 2)),
                           (slice(None), slice(1, None, 2))]

        la = load_npy(self.output_path, dim_datas[self.rank], comm=self.comm)
        assert_equal(la, self.expected[expected_slices[self.rank]])

    def test_load_nu(self):
        dim_datas = nu_test_data
        expected_indices = [dd[1]['indices'] for dd in dim_datas]

        la = load_npy(self.output_path, dim_datas[self.rank], comm=self.comm)
        assert_equal(la, self.expected[:, expected_indices[self.rank]])


class TestHdf5FileSave(MpiTestCase):

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
        la = LocalArray((51,), comm=self.comm)
        np_arr = numpy.random.random(la.local_shape)
        la.set_localarray(np_arr)
        save_hdf5(self.output_path, la, key=self.key, mode='w')

        # check saved file
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            for i, v in ndenumerate(la):
                self.assertEqual(v, fp[self.key][i])

    def test_save_2d(self):
        la = LocalArray((11, 15), comm=self.comm)
        np_arr = numpy.random.random(la.local_shape)
        la.set_localarray(np_arr)
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


class TestHdf5FileLoad(MpiTestCase):

    @classmethod
    def get_comm_size(cls):
        return 2

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
        dim_datas = bn_test_data
        la = load_hdf5(self.output_path, dim_datas[self.rank],
                       key=self.key, comm=self.comm)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            assert_equal(la, self.expected[numpy.newaxis, self.rank])

    def test_load_nc(self):
        dim_datas = nc_test_data
        expected_slices = [(slice(None), slice(0, None, 2)),
                           (slice(None), slice(1, None, 2))]
        la = load_hdf5(self.output_path, dim_datas[self.rank],
                       key=self.key, comm=self.comm)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            expected_slice = expected_slices[self.rank]
            assert_equal(la, self.expected[expected_slice])

    def test_load_nu(self):
        dim_datas = nu_test_data
        expected_indices = [dd[1]['indices'] for dd in dim_datas]
        la = load_hdf5(self.output_path, dim_datas[self.rank],
                       key=self.key, comm=self.comm)
        with self.h5py.File(self.output_path, 'r', driver='mpio',
                            comm=self.comm) as fp:
            assert_equal(la, self.expected[:, expected_indices[self.rank]])
