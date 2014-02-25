"""
Tests for distributed IO.

Many of these tests require a 4-engine cluster to be running locally, and will
write out temporary files.
"""

import unittest
import tempfile

import numpy as np
from numpy.testing import assert_equal

from os import path
from six.moves import range

from IPython.parallel import Client
from distarray.client import Context, DistArray


class TestDistributedIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.dv = cls.client[:]
        if len(cls.dv.targets) < 4:
            errmsg = 'Must set up a cluster with at least 4 engines running.'
            raise unittest.SkipTest(errmsg)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def tearDown(self):
        self.dv.clear()

    def test_flat_file_read_write(self):
        dac = Context(self.dv)
        da = dac.empty((100,), dist={0: 'b'})

        output_dir = tempfile.gettempdir()
        filename = 'outfile'
        output_path = path.join(output_dir, filename)
        dac.save(output_path, da)
        db = dac.load(output_path)
        self.assertTrue(isinstance(db, DistArray))
        self.assertEqual(da, db)

    def test_hdf5_file_write_block(self):
        import h5py

        datalen = 33
        dac = Context(self.dv)
        da = dac.empty((datalen,), dist={0: 'b'})
        for i in range(datalen):
            da[i] = i

        output_dir = tempfile.gettempdir()
        filename = 'outfile.hdf5'
        output_path = path.join(output_dir, filename)
        dac.save_hdf5(output_path, da)

        self.assertTrue(path.exists(output_path))
        fp = h5py.File(output_path, 'r')
        self.assertTrue("buffer" in fp)

        expected = np.arange(datalen)
        assert_equal(expected, fp["buffer"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
