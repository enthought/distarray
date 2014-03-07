# encoding: utf-8 ( #135 )
__docformat__ = "restructuredtext en"
# Copyright (c) 2008-2014, IPython Development Team and Enthought, Inc.

import unittest
import numpy
from numpy.testing import assert_array_equal

import distarray.local.denselocalarray as dla
from distarray.error import InvalidCommSizeError, InvalidRankError
from distarray.mpiutils import MPI, create_comm_of_size, create_comm_with_list
from distarray.testing import comm_null_passes


class TestCreateCommWithList(unittest.TestCase):
    """ Test creating MPI comm with comm_with_list().

    Note that this is not derived from the usual MpiTestCase.
    This is so that this can create the MPI communicator using
    an alternate code path that is otherwise not tested.
    """

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


class TestCreateInvalidComm(unittest.TestCase):
    """ Test that invalid MPI comm creation fails as expected.

    Note that this is not derived from the usual MpiTestCase.
    This is so that we can exercise some failure cases, which
    are not triggered by MpiTestCase since it works properly.
    """

    def setUp(self):
        # Get the maximum number of nodes available.
        COMM_PRIVATE = MPI.COMM_WORLD.Clone()
        self.max_size = COMM_PRIVATE.Get_size()

    def test_size_too_big(self):
        """ Test that a comm of size with too many nodes will fail. """
        too_many = 2 * self.max_size
        with self.assertRaises(InvalidCommSizeError):
            create_comm_of_size(too_many)

    def test_list_too_big(self):
        """ Test that a comm from list with too many nodes will fail. """
        too_many = 2 * self.max_size
        nodes = range(too_many)
        with self.assertRaises(InvalidCommSizeError):
            create_comm_with_list(nodes)

    def test_invalid_ranks(self):
        """ Test that a comm from list with invalid ranks will fail. """
        max_size = self.max_size
        # Nodes should be 0..max_size - 1, so create invalid values.
        nodes = range(10, max_size + 10)
        with self.assertRaises(InvalidRankError):
            create_comm_with_list(nodes)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
