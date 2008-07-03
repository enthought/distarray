# Includes from the python headers

# Include the Numpy C API for use via Cython extension code
cimport numpy as cnumpy

from distarray.error import DistArrayError
from distarray.core import densedistarray as ddarray

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
	
	enum: FFTW_FORWARD
	enum: FFTW_BACKWARD

	enum: FFTW_ESTIMATE

	ctypedef struct fftw_complex:
		pass

	ctypedef struct fftwf_complex:
		pass
	
	ctypedef struct fftw_plan:
		pass

	
	void* fftw_malloc(size_t n)

	void fftw_destroy_plan(fftw_plan p)
	
	void fftw_execute(fftw_plan p)

	fftw_plan fftwf_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1, fftwf_complex* inArray, 
			fftwf_complex* outArray, cmpi.MPI_Comm comm, int sign, unsigned flags)

	fftw_plan fftw_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1, fftw_complex* inArray, 
			fftw_complex* outArray, cmpi.MPI_Comm comm, int sign, unsigned flags)


	fftw_plan fftwf_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1, float* inArray, 
			fftwf_complex* outArray, cmpi.MPI_Comm comm, unsigned flags)


	fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1, double* inArray, 
			fftw_complex* outArray, cmpi.MPI_Comm comm, unsigned flags)




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

class DistArrayFFTError(DistArrayError):
	pass

def py_fftw_mpi_init():
	fftw_mpi_init()
	
def py_fftw_mpi_cleanup():
	fftw_mpi_cleanup()

def py_fftw_mpi_plan_dft_r2c_2d(n0, n1, cnumpy.ndarray input, cnumpy.ndarray output, CommType mpicom, flags):
	if (input.shape != output.shape):
		raise DistArrayFFTError("Input and Output shape must match for fftw r2c dft 2d")
	if (input.ndim != 2):
		raise DistArrayFFTError("Input and Output must be 2-dimensional for fftw r2c dft 2d")

	if not (input.flags.c_contiguous and output.flags.c_contiguous and input.flags.aligned and output.flags.aligned):
		raise DistArrayFFTError, "FFTW requires that Input and Output arrays must be aligned and contiguous"

	cdef py_fftw_plan plan = py_fftw_plan()
	if (input.dtype == numpy.float64 and output.dtype == numpy.complex128):
		plan.p = fftw_mpi_plan_dft_r2c_2d(n0, n1, <double*>input.c_data , <fftw_complex*>output.c_data , mpicom.ob_mpi, flags)
	elif (input.dtype == numpy.float32 and output.dtype == numpy.complex64):
		plan.p = fftwf_mpi_plan_dft_r2c_2d(n0, n1, <float*>input.c_data , <fftwf_complex*>output.c_data , mpicom.ob_mpi, flags)
	else:
		raise DistArrayFFTError("Input and Output data type must be float32/64 and complex64/128 respectively for fftw r2c dft")
		return None
	return plan


def py_fftw_mpi_plan_dft_2d(n0, n1, cnumpy.ndarray input, cnumpy.ndarray output, CommType mpicom, sign, flags):
	# sanity check on in/out shape, make sure complex, contiguous, aligned
	if (input.shape != output.shape):
		raise DistArrayFFTError("Input and Output shape must match for fftw (c2c) dft")
	if (input.ndim != 2):
		raise DistArrayFFTError("Input and Output must be 2-dimensional for fftw (c2c) dft 2d")

	if not (input.flags.c_contiguous and output.flags.c_contiguous and input.flags.aligned and output.flags.aligned):
		raise DistArrayFFTError, "FFTW requires that Input and Output arrays must be aligned and contiguous"

	cdef py_fftw_plan plan = py_fftw_plan()
	# sign --> FFTW_FORWARD = -1
	# flags --> FFTW_ESTIMATE = 2^6 = 64
	if (input.dtype == output.dtype == numpy.complex128):
		plan.p = fftw_mpi_plan_dft_2d(n0, n1, <fftw_complex*>input.c_data , <fftw_complex*>output.c_data , mpicom.ob_mpi, sign, flags)
	elif (input.dtype == output.dtype == numpy.complex64):
		plan.p = fftwf_mpi_plan_dft_2d(n0, n1, <fftwf_complex*>input.c_data , <fftwf_complex*>output.c_data , mpicom.ob_mpi, sign, flags)
	else:
		raise DistArrayFFTError("Input and Output data type must be complex64/128 for fftw (c2c) dft")
		return None
	return plan
	
def py_fftw_mpi_local_size_2d(n0, n1, CommType mpicom):
	cdef ptrdiff_t ln0, ls0
	localsize = fftw_mpi_local_size_2d(n0, n1, mpicom.ob_mpi, &ln0, &ls0)
	return (localsize, ln0, ls0)

def py_fftw_execute(py_fftw_plan p):
	fftw_execute(p.p)

def py_fftw_destroy_plan(py_fftw_plan p):
	fftw_destroy_plan(p.p)


###### BEGING PYTHON (public) INTERFACES #####

def fft():
	pass
		
def ifft():
	pass


def fft2(A):
	if A.ndim != 2:
		raise DistArrayFFTError("fft2 must be called with a 2-dimensional array.")
	(x,y) = A.shape

	if (A.ndistdim != 1):
		raise DistArrayFFTError("fftw requires that only the first dimension be decomposed.  The number of decomposed \
			dimensions is other than one")
	if (A.distdims != (0,) ):
		raise DistArrayFFTError("fftw requires that only the first dimension be decomposed.  The decomposed dimension is not the first")
	if (A.dist != ('b',None) ):
		raise DistArrayFFTError("fftw requires block decomposition.  The decomposition is other than block")
	
	(lx,ly) = A.local_view().shape
	(localsize, ln0, ls0) = py_fftw_mpi_local_size_2d(x,y, A.comm)
	if (lx*ly != localsize or ln0 != lx):
		raise DistArrayFFTError("Distarray decomposition distribution does not match FFTW's expectations (local size is wrong)")




	retval = None

	if (A.dtype==numpy.complex64 or A.dtype==numpy.complex128):
		retval = ddarray.empty_like(A)
		pln = py_fftw_mpi_plan_dft_2d(x,y, A.local_view(), retval.local_view(), A.comm, FFTW_FORWARD, FFTW_ESTIMATE)
		py_fftw_execute(pln)
		py_fftw_destroy_plan(pln)
		# del pln
		return retval

	elif numpy.issubdtype(A.dtype, numpy.complexfloating):#in-place of new-typed copy (unknown complex)
		retval = A.astype(numpy.complex128)
		pln = py_fftw_mpi_plan_dft_2d(x,y, retval.local_view(), retval.local_view(), A.comm, FFTW_FORWARD, FFTW_ESTIMATE)
		py_fftw_execute(pln)
		py_fftw_destroy_plan(pln)
		return retval
	else:
		tq32 = (A.dtype==numpy.float32)
		tq64 = (A.dtype==numpy.float64)
		if tq64:
			retval = ddarray.empty_like(A, numpy.complex128)
		elif tq32:
			retval = ddarray.empty_like(A, numpy.complex64)
		if (tq64 or tq32):
			pln = py_fftw_mpi_plan_dft_r2c_2d(x,y, A.local_view(), retval.local_view(), A.comm, FFTW_ESTIMATE)
			py_fftw_execute(pln)
			py_fftw_destroy_plan(pln)
			# del pln
			return retval
		elif numpy.issubdtype(A.dtype, numpy.integer) or numpy.issubdtype(A.dtype, numpy.floating): #int or unknown floating
			retval = A.astype(numpy.complex128)
			pln = py_fftw_mpi_plan_dft_2d(x,y, retval.local_view(), retval.local_view(), retval.comm, FFTW_FORWARD, FFTW_ESTIMATE)
			py_fftw_execute(pln)
			py_fftw_destroy_plan(pln)
			return retval
		else:
			raise DistArrayFFTError("ifft2 called with unsupported type: " + str(A.dtype))




def ifft2(A):
	if A.ndim != 2:
		raise DistArrayFFTError("ifft2 must be called with a 2-dimensional array.")
	(x,y) = A.shape

	if (A.ndistdim != 1):
		raise DistArrayFFTError("fftw requires that only the first dimension be decomposed.  The number of decomposed \
			dimensions is other than one")
	if (A.distdims != (0,) ):
		raise DistArrayFFTError("fftw requires that only the first dimension be decomposed.  The decomposed dimension is not the first")
	if (A.dist != ('b',None) ):
		raise DistArrayFFTError("fftw requires block decomposition.  The decomposition is other than block")
	
	(lx,ly) = A.local_view().shape
	(localsize, ln0, ls0) = py_fftw_mpi_local_size_2d(x,y, A.comm)
	if (lx*ly != localsize or ln0 != lx):
		raise DistArrayFFTError("Distarray decomposition distribution does not match FFTW's expectations (local size is wrong)")

	if numpy.issubdtype(A.dtype, numpy.complexfloating) or numpy.issubdtype(A.dtype, numpy.integer) or numpy.issubdtype(A.dtype, numpy.floating):
		if (A.dtype==numpy.complex64 or A.dtype==numpy.complex128):
			retval = ddarray.empty_like(A)
			pln = py_fftw_mpi_plan_dft_2d(x,y, A.local_view(), retval.local_view(), A.comm, FFTW_BACKWARD, FFTW_ESTIMATE)
			py_fftw_execute(pln)
			py_fftw_destroy_plan(pln)
			# del pln
			ddarray.divide(retval, (x*y), retval)
			return retval
		else:
			retval = A.astype(numpy.complex128)
			pln = py_fftw_mpi_plan_dft_2d(x,y, retval.local_view(), retval.local_view(), A.comm, FFTW_BACKWARD, FFTW_ESTIMATE)
			py_fftw_execute(pln)
			py_fftw_destroy_plan(pln)
			ddarray.divide(retval, (x*y), retval)
			return retval
	else:
		raise DistArrayFFTError("ifft2 called with unsupported type: " + str(A.dtype))



def fftn():
	pass

def ifftn():
	pass

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

py_fftw_mpi_init()
import atexit
atexit.register(py_fftw_mpi_cleanup)

