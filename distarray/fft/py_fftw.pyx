# Includes from the python headers

# Include the Numpy C API for use via Cython extension code
cimport numpy as cnumpy

from distarray.error import DistArrayError

# include "mpi4py/mpi.pxi"

from mpi4py.MPI cimport Comm as CommType
cimport mpi4py.mpi_c as cmpi

################################################
# Initialize numpy - this MUST be done before any other code is executed.
cnumpy.import_array()

import numpy


###### STD C EXTERNS #######

cdef extern from "stddef.h":
	ctypedef long size_t
	ctypedef long ptrdiff_t

###### FFTW C EXTERNS ######
	
cdef extern from "fftw3-mpi.h":
	void fftw_mpi_init()
	void fftw_mpi_cleanup()
	
	ctypedef struct fftw_complex:
		pass
	
	ctypedef struct fftw_plan:
		pass

	
	void* fftw_malloc(size_t n)

	void fftw_destroy_plan(fftw_plan p)
	
	void fftw_execute(fftw_plan p)

	fftw_plan fftw_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1, fftw_complex* inArray, 
			fftw_complex* outArray, cmpi.MPI_Comm comm, int sign, unsigned flags)	

	ptrdiff_t fftw_mpi_local_size_2d(ptrdiff_t n0, ptrdiff_t n1, cmpi.MPI_Comm comm,
			ptrdiff_t *local_n0, ptrdiff_t *local_0_start)	
	
	ptrdiff_t fftw_mpi_local_size_many(int rank, ptrdiff_t *n,
			ptrdiff_t howmany, ptrdiff_t block0, cmpi.MPI_Comm comm, ptrdiff_t *local_n0,
			ptrdiff_t *local_0_start)

	ptrdiff_t fftw_mpi_local_size_1d(ptrdiff_t n0, cmpi.MPI_Comm comm, int sign, 
			unsigned flags, ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
			ptrdiff_t *local_no, ptrdiff_t *local_o_start)
	

######  END EXTERNALS  ######

######  BEGIN PYTHON (internal) INTERFACES #####
	
cdef class py_fftw_plan:
	cdef fftw_plan p

class DistArray_fftw_Error(DistArrayError):
	pass

def py_fftw_mpi_init():
	fftw_mpi_init()
	
def py_fftw_mpi_cleanup():
	fftw_mpi_cleanup()

def py_fftw_mpi_plan_dft_2d(n0, n1, cnumpy.ndarray input, cnumpy.ndarray output, CommType mpicom, sign, flags):
	# sanity check on in/out shape, make sure complex, contiguous, aligned
	if (input.shape != output.shape):
		raise DistArray_fftw_Error, "Input and Output shape must match for fftw c2c dft"

	if not (input.dtype == output.dtype == numpy.complex128):
		raise DistArray_fftw_Error, "Input and Output data type must be complex128 for fftw c2c dft"

	if not (input.flags.c_contiguous and output.flags.c_contiguous and input.flags.aligned and output.flags.aligned):
		raise DistArray_fftw_Error, "FFTW requires that Input and Output arrays must be aligned and contiguous"

	cdef py_fftw_plan plan = py_fftw_plan()

	# sign --> FFTW_FORWARD = -1
	# flags --> FFTW_ESTIMATE = 2^6 = 64
	plan.p = fftw_mpi_plan_dft_2d(n0, n1, <fftw_complex*>input.c_data , <fftw_complex*>output.c_data , mpicom.ob_mpi, sign, flags)
	return plan
	

def py_fftw_execute(py_fftw_plan p):
	fftw_execute(p.p)

def py_fftw_destroy_plan(py_fftw_plan p):
	fftw_destroy_plan(p.p)


###### BEGING PYTHON (public) INTERFACES #####

def fft():
	pass

def fft2(A):


def ifft():
	pass
def ifff2():
	

def rfft():
	pass
def rfft2():
	pass
def irfft():
	pass
def irfft2():
	pass

### TODO
#[] # SANITY CHECK ON DATA DISTRIBUTION (OVER PROCESSES)

#[x] #SANITY CHECK ON INPUT VS DESTINATION SHAPE/TYPE

# <fftw_plan> CACHE AND APPROPRIATE CHECKS.
