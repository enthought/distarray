"""
Tests for distributed IO.

Many of these tests require a 4-engine cluster to be running locally, and will
write out temporary files.
"""

import unittest
import tempfile
from os import path
from IPython.parallel import Client
from distarray.client import DistArray
from distarray.context import Context


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
        dac = Context(self.client)
        da = dac.empty((100,), dist={0: 'b'})

        output_dir = tempfile.gettempdir()
        filename = 'outfile'
        output_path = path.join(output_dir, filename)
        dac.save(output_path, da)
        db = dac.load(output_path)
        self.assertTrue(isinstance(db, DistArray))
        self.assertEqual(da, db)


if __name__ == '__main__':
    unittest.main(verbosity=2)
