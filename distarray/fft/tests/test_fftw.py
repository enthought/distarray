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
from distarray.core import maps_fast as maps, densedistarray
from distarray import utils
from distarray.fft import py_fftw as fftw

class Test2dForward(unittest.TestCase):

    def test_float64(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16),comm=comm,dtype='float64')
            except NullCommError:
                pass
            else:
                result = fftw.fft2(da)
                numpy_result = np.fft.fft2(np.fromfunction(f,(16,16),dtype='float64'))
                (low, high) = result.global_limits(0)
                assert_array_almost_equal(result.local_view(), numpy_result[low:high+1,:], 6)
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()

    def test_float32(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16),comm=comm,dtype='float32')
            except NullCommError:
                pass
            else:
                result = fftw.fft2(da)
                numpy_result = np.fft.fft2(np.fromfunction(f,(16,16),dtype='float32'))
                (low, high) = result.global_limits(0)
                assert_array_almost_equal(result.local_view(), numpy_result[low:high+1,:], 4)
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()

    def test_int16(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16),comm=comm,dtype='int16')
            except NullCommError:
                pass
            else:
                result = fftw.fft2(da)
                numpy_result = np.fft.fft2(np.fromfunction(f,(16,16),dtype='int16'))
                (low, high) = result.global_limits(0)
                assert_array_almost_equal(result.local_view(), numpy_result[low:high+1,:], 6)
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()

    def test_complex128(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16), dist=('b',None),comm=comm,dtype='complex128')
            except NullCommError:
                pass
            else:
                result = fftw.fft2(da)
                numpy_result = np.fft.fft2(np.fromfunction(f,(16,16),dtype='complex128'))
                (low, high) = result.global_limits(0)
                assert_array_almost_equal(result.local_view(), numpy_result[low:high+1,:], 6)
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()
    def test_complex128_input(self):
        try:
            comm = create_comm_of_size(4)
        except InvalidCommSizeError:
            pass
        else:
            try:
                def f(i,j):
                    return i+j
                da = densedistarray.fromfunction(f,(16,16), dist=('b',None),comm=comm,dtype='complex128')
            except NullCommError:
                pass
            else:
                result = fftw.fft2(da)
                numpy_input = np.fromfunction(f,(16,16),dtype='complex128')
                numpy_result = np.fft.fft2(numpy_input)
                (low, high) = result.global_limits(0)
                assert_array_equal(da.local_view(), numpy_input[low:high+1,:])
                # Compare the part of numpy_result that this processor has with result.local_array
                comm.Free()


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass