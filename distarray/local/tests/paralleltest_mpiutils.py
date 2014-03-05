import unittest
import numpy
from numpy.testing import assert_array_equal

import distarray.local.denselocalarray as dla
from distarray.testing import comm_null_passes
from distarray.mpiutils import MPI, create_comm_with_list


class TestCreateCommWithList(unittest.TestCase):
    """ Test creating MPI comm with comm_with_list(). """

    def setUp(self):
        self.nodes = [0, 1, 2, 3]
        self.comm = create_comm_with_list(self.nodes)

    def tearDown(self):
        if self.comm != MPI.COMM_NULL:
            self.comm.Free()

    @comm_null_passes
    def test_zeros(self):
        """ A very simple test that really checks the MPI comm. """
        size = len(self.nodes)
        nrows = size * 3
        a = dla.zeros((nrows, 20), comm=self.comm)
        expected = numpy.zeros((nrows // size, 20))
        assert_array_equal(a.local_array, expected)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
