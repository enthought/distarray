# encoding: utf-8 ( #135 )
__docformat__ = "restructuredtext en"
# Copyright (c) 2008-2014, IPython Development Team and Enthought, Inc.

import unittest
import numpy

from distarray.mpiutils import MPI, mpi_type_for_ndarray


class TestMpiTypes(unittest.TestCase):
    """ Test the mpi_type_for_ndarray method. """

    def subtest_mpi_type_for_ndarray(self, np_dtype, expected_mpi_dtype):
        """ Test that the MPI type for a numpy array is as expected. """
        arr = numpy.ones((3, 3), dtype=np_dtype)
        actual_mpi_dtype = mpi_type_for_ndarray(arr)
        self.assertEqual(actual_mpi_dtype, expected_mpi_dtype)

    def test_mpi_type_for_ndarray(self):
        """ Test the set of allowed numpy types. """
        # 'f'
        np_dtype, mpi_dtype = numpy.dtype('f'), MPI.FLOAT
        self.subtest_mpi_type_for_ndarray(np_dtype, mpi_dtype)
        # 'd'
        np_dtype, mpi_dtype = numpy.dtype('d'), MPI.DOUBLE
        self.subtest_mpi_type_for_ndarray(np_dtype, mpi_dtype)
        # 'i'
        np_dtype, mpi_dtype = numpy.dtype('i'), MPI.INTEGER
        self.subtest_mpi_type_for_ndarray(np_dtype, mpi_dtype)
        # 'l'
        np_dtype, mpi_dtype = numpy.dtype('l'), MPI.LONG
        self.subtest_mpi_type_for_ndarray(np_dtype, mpi_dtype)


if __name__ == '__main__':
    unittest.main(verbosity=2)
