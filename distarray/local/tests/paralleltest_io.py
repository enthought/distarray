# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

import tempfile
import os
from numpy.testing import assert_allclose
from distarray.local import LocalArray, save, load, save_hdf5
from distarray.testing import MpiTestCase, import_or_skip, temp_filepath


class TestFlatFileIO(MpiTestCase):

    def setUp(self):
        self.larr0 = LocalArray((7,), comm=self.comm)
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


class TestHDF5FileIO(MpiTestCase):

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
