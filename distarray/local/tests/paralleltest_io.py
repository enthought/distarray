import unittest
import tempfile
from numpy.testing import assert_allclose
from os import path
from distarray.local import LocalArray, save, load
from distarray.testing import comm_null_passes, MpiTestCase


class TestFlatFileIO(MpiTestCase):

    @comm_null_passes
    def test_flat_file_read_write(self):
        larr0 = LocalArray((7,))
        output_dir = tempfile.gettempdir()
        filename = 'outfile'
        output_path = path.join(output_dir, filename)
        save(output_path, larr0)
        larr1 = load(output_path, comm=self.comm)
        self.assertTrue(isinstance(larr1, LocalArray))
        assert_allclose(larr0, larr1)
