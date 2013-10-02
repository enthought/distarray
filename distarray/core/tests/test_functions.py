import unittest
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from distarray.core.error import *
from distarray.mpi.error import *
from distarray.mpi import mpibase
from distarray.mpi.mpibase import (
    MPI,
    create_comm_of_size,
    create_comm_with_list)
from distarray.core import maps, denselocalarray


class TestFunctions(unittest.TestCase):

    def test_arecompatible(self):
        """
        Test if two DistArrays are compatible.
        """
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.LocalArray((16,16), dtype='int64', comm=comm)
                b = denselocalarray.LocalArray((16,16), dtype='float32', comm=comm)
            except NullCommError:
                raise unittest.SkipTest("Skipped due to Null Comm")
            else:
                self.assertEqual(denselocalarray.arecompatible(a,b), True)
                a = denselocalarray.LocalArray((16,16), dtype='int64', dist='c', comm=comm)
                b = denselocalarray.LocalArray((16,16), dtype='float32', dist='b', comm=comm)
                self.assertEqual(denselocalarray.arecompatible(a,b), False)
                comm.Free()

    def test_fromfunction(self):
        """
        Can we build an array using fromfunction and a trivial function.
        """
        def f(*global_inds):
            return 1.0

        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.fromfunction(f, (16,16), dtype='int64', dist=('b','c'), comm=comm)
            except NullCommError:
                raise unittest.SkipTest("Skipped due to Null Comm")
            else:
                self.assertEqual(a.shape, (16,16))
                self.assertEqual(a.dtype, np.dtype('int64'))
                for global_inds, value in denselocalarray.ndenumerate(a):
                    self.assertEqual(1.0, value)
                comm.Free()

    def test_fromfunction_complicated(self):
        """
        Can we build an array using fromfunction and a nontrivial function.
        """
        def f(*global_inds):
            return sum(global_inds)

        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            raise unittest.SkipTest("Skipped due to Invalid Comm Size")
        else:
            try:
                a = denselocalarray.fromfunction(f, (16,16), dtype='int64', dist=('b','c'), comm=comm)
            except NullCommError:
                raise unittest.SkipTest("Skipped due to Null Comm")
            else:
                self.assertEqual(a.shape, (16,16))
                self.assertEqual(a.dtype, np.dtype('int64'))
                for global_inds, value in denselocalarray.ndenumerate(a):
                    self.assertEqual(sum(global_inds), value)
                comm.Free()


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
