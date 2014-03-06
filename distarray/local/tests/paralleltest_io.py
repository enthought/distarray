import tempfile
import os
import numpy

from numpy.testing import assert_allclose, assert_equal
from distarray.local import LocalArray, save, load, save_hdf5, load_hdf5
from distarray.testing import (comm_null_passes, MpiTestCase, import_or_skip,
                               temp_filepath)


class TestFlatFileIO(MpiTestCase):

    @comm_null_passes
    def more_setUp(self):
        self.larr0 = LocalArray((7,), comm=self.comm)
        self.output_path = temp_filepath(extension='.dnpy')

    @comm_null_passes
    def more_tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    @comm_null_passes
    def test_flat_file_save_with_filename(self):
        save(self.output_path, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    @comm_null_passes
    def test_flat_file_save_with_file_object(self):
        with open(self.output_path, 'wb') as fp:
            save(fp, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    @comm_null_passes
    def test_flat_file_save_load_with_filename(self):
        save(self.output_path, self.larr0)
        larr1 = load(self.output_path, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)

    @comm_null_passes
    def test_flat_file_save_load_with_file_object(self):
        save(self.output_path, self.larr0)
        with open(self.output_path, 'rb') as fp:
            larr1 = load(fp, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)


class TestHDF5FileIO(MpiTestCase):

    def get_comm_size(self):
        return 2

    @comm_null_passes
    def test_save(self):
        h5py = import_or_skip('h5py')

        key = "data"
        larr0 = LocalArray((51,), comm=self.comm)
        output_dir = tempfile.gettempdir()
        filename = 'localarray_hdf5_save_test.hdf5'
        output_path = os.path.join(output_dir, filename)
        try:
            save_hdf5(output_path, larr0, key=key, mode='w')

            if self.comm.Get_rank() == 0:
                with h5py.File(output_path, 'r') as fp:
                    self.assertTrue("data" in fp)
        finally:
            if self.comm.Get_rank() == 0:
                if os.path.exists(output_path):
                    os.remove(output_path)

    @comm_null_passes
    def test_load_bn(self):
        h5py = import_or_skip('h5py')

        output_dir = tempfile.gettempdir()
        filename = 'localarray_hdf5_load_test_bn.hdf5'
        output_path = os.path.join(output_dir, filename)

        dim_data_0 = (
            {'size': 2,
             'dist_type': 'b',
             'proc_grid_rank': 0,
             'proc_grid_size': 2,
             'start': 0,
             'stop': 1,
            },
            {'size': 10,
             'dist_type': 'n',
            })

        dim_data_1 = (
            {'size': 2,
             'dist_type': 'b',
             'proc_grid_rank': 1,
             'proc_grid_size': 2,
             'start': 1,
             'stop': 2,
            },
            {'size': 10,
             'dist_type': 'n',
            })

        dim_datas = [dim_data_0, dim_data_1]
        expected = numpy.arange(20).reshape(2, 10)

        if self.comm.Get_rank() == 0:
            with h5py.File(output_path, 'w') as fp:
                fp["load_test"] = expected
        self.comm.Barrier() # wait until file exists

        try:
            la = load_hdf5(output_path, dim_datas[self.comm.Get_rank()],
                           key="load_test", comm=self.comm)
            with h5py.File(output_path, 'r') as fp:
                assert_equal(la, expected[numpy.newaxis, self.comm.Get_rank()])
        finally:
            if self.comm.Get_rank() == 0:
                if os.path.exists(output_path):
                    os.remove(output_path)


    @comm_null_passes
    def test_load_nc(self):
        h5py = import_or_skip('h5py')

        output_dir = tempfile.gettempdir()
        filename = 'localarray_hdf5_load_test_nc.hdf5'
        output_path = os.path.join(output_dir, filename)

        dim_data_0 = (
            {'size': 10,
             'dist_type': 'n',
            },
            {'size': 2,
             'dist_type': 'c',
             'proc_grid_rank': 0,
             'proc_grid_size': 2,
             'start': 0,
            },
            )

        dim_data_1 = (
            {'size': 10,
             'dist_type': 'n',
            },
            {'size': 2,
             'dist_type': 'c',
             'proc_grid_rank': 1,
             'proc_grid_size': 2,
             'start': 1,
            },
            )

        dim_datas = [dim_data_0, dim_data_1]
        expected = numpy.arange(20).reshape(2, 10)
        expected_slices = [(slice(None), slice(0, None, 2)),
                           (slice(None), slice(1, None, 2))]

        if self.comm.Get_rank() == 0:
            with h5py.File(output_path, 'w') as fp:
                fp["load_test"] = expected
        self.comm.Barrier() # wait until file exists

        try:
            la = load_hdf5(output_path, dim_datas[self.comm.Get_rank()],
                           key="load_test", comm=self.comm)
            with h5py.File(output_path, 'r') as fp:
                expected_slice = expected_slices[self.comm.Get_rank()]
                assert_equal(la, expected[expected_slice])
        finally:
            if self.comm.Get_rank() == 0:
                if os.path.exists(output_path):
                    os.remove(output_path)

    @comm_null_passes
    def test_load_u(self):
        h5py = import_or_skip('h5py')

        output_dir = tempfile.gettempdir()
        filename = 'localarray_hdf5_load_test_u.hdf5'
        output_path = os.path.join(output_dir, filename)

        # Note: indices must be in increasing order
        #       (restiction of h5py / HDF5)

        dim_data_0 = (
            {'size': 20,
             'dist_type': 'u',
             'proc_grid_rank': 0,
             'proc_grid_size': 2,
             'indices': [0, 3, 4, 6, 8, 10, 13, 15, 18],
            },
            )

        dim_data_1 = (
            {'size': 20,
             'dist_type': 'u',
             'proc_grid_rank': 1,
             'proc_grid_size': 2,
             'indices': [1, 2, 5, 7, 9, 11, 12, 14, 16, 17, 19],
            },
            )

        dim_datas = [dim_data_0, dim_data_1]
        expected = numpy.arange(20)
        expected_indices = [dd[0]['indices'] for dd in dim_datas]

        if self.comm.Get_rank() == 0:
            with h5py.File(output_path, 'w') as fp:
                fp["load_test"] = expected
        self.comm.Barrier() # wait until file exists

        try:
            rank = self.comm.Get_rank()
            la = load_hdf5(output_path, dim_datas[rank],
                           key="load_test", comm=self.comm)
            with h5py.File(output_path, 'r') as fp:
                expected_index = expected_indices[rank]
                assert_equal(la, expected[expected_index])
        finally:
            if self.comm.Get_rank() == 0:
                if os.path.exists(output_path):
                    os.remove(output_path)
