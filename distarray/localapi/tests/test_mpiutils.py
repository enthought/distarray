# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
import numpy

from distarray.localapi.mpiutils import mpi_type_for_ndarray


class TestMpiTypes(unittest.TestCase):
    """ Test the mpi_type_for_ndarray method. """

    def test_mpi_type_for_ndarray(self):
        """ Test the mpi_type_for_ndarray method. """
        arr = numpy.ones((3, 3))
        mpi_dtype = mpi_type_for_ndarray(arr)
        # The specific values returned are beyond the scope of this project,
        # so just check that we get something.
        self.assertIsNotNone(mpi_dtype)


if __name__ == '__main__':
    unittest.main(verbosity=2)
