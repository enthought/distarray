import tempfile
import os
from numpy.testing import assert_allclose
from distarray.local import LocalArray, save, load, save_hdf5
from distarray.testing import (comm_null_passes, MpiTestCase, import_or_skip,
                               temp_filepath)


class TestFlatFileIO(MpiTestCase):

    @comm_null_passes
    def test_flat_file_read_write(self):
        larr0 = LocalArray((7,), comm=self.comm)
        output_dir = tempfile.gettempdir()
        filename_prefix = "localarray_flatfiletest"
        output_path = os.path.join(output_dir, filename_prefix)
        try:
            save(output_path, larr0)
            larr1 = load(output_path, comm=self.comm)
            self.assertTrue(isinstance(larr1, LocalArray))
            assert_allclose(larr0, larr1)
        finally:
            rank = str(self.comm.Get_rank())
            outfile = output_path + "_" + rank + ".dnpy"
            if os.path.exists(outfile):
                os.remove(outfile)


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
