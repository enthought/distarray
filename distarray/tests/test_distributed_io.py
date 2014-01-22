"""
Tests for distributed IO.

Many of these tests require a 4-engine cluster to be running locally, and will
write out temporary files.
"""

import unittest
import tempfile
from os import path
from IPython.parallel import Client
from distarray.client import Context, DistArray


class TestDistributedIO(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        if len(self.dv.targets) < 4:
            errmsg = 'Must set up a cluster with at least 4 engines running.'
            raise unittest.SkipTest(errmsg)
        self.dac = Context(self.dv)
        self.da = self.dac.empty((100,), dist={0: 'b'})

    def test_flat_file_read_write(self):
        output_dir = tempfile.gettempdir()
        filename = 'outfile'
        output_path = path.join(output_dir, filename)
        self.dac.save(output_path, self.da)
        self.db = self.dac.load(output_path)
        self.assertTrue(isinstance(self.db, DistArray))


if __name__ == '__main__':
    unittest.main(verbosity=2)
