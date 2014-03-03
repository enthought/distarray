import tempfile
import os
from numpy.testing import assert_allclose
from distarray.local import LocalArray, save, load, save_hdf5
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
    def test_flat_file_save_with_file(self):
        with open(self.output_path, 'wb') as fp:
            save(fp, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    @comm_null_passes
    def test_flat_file_save_with_full_filename(self):
        save(self.output_path, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    @comm_null_passes
    def test_flat_file_save_with_no_ext_filename(self):
        output_path = self.output_path.replace('.dnpy', '')
        save(self.output_path, self.larr0)

        with open(self.output_path, 'rb') as fp:
            magic = fp.read(6)

        self.assertTrue(magic == b'\x93DARRY')

    @comm_null_passes
    def test_flat_file_save_load_with_full_filename(self):
        save(self.output_path, self.larr0)
        larr1 = load(self.output_path, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)

    @comm_null_passes
    def test_flat_file_save_load_with_file(self):
        save(self.output_path, self.larr0)
        with open(self.output_path, 'rb') as fp:
            larr1 = load(fp, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(self.larr0, larr1)


class TestHDF5FileIO(MpiTestCase):

    @comm_null_passes
    def test_hdf5_file_write(self):
        h5py = import_or_skip('h5py')

        key = "data"
        larr0 = LocalArray((51,), comm=self.comm)
        output_dir = tempfile.gettempdir()
        filename = 'localarray_hdf5test.hdf5'
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
