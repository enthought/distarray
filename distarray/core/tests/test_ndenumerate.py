from __future__ import print_function
import unittest
from distarray.core import denselocalarray
from distarray.mpi.error import InvalidCommSizeError
from distarray.mpi.mpibase import create_comm_of_size
from distarray.core.error import NullCommError


class TestNDEnumerate(unittest.TestCase):
    """Make sure we generate indices compatible with __getitem__."""

    def test_ndenumerate(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16, 16, 2),
                                               dist=('c', 'b', None),
                                               comm=comm)
            except NullCommError:
                raise unittest.SkipTest("Skipped due to Null Comm")
            else:
                for global_inds, value in denselocalarray.ndenumerate(a):
                    a[global_inds] = 0.0
                comm.Free()


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
