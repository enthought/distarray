import distarray
from distarray.fft import py_fftw

def funcy(i,j):
	return i+j

#A = distarray.fromfunction(funcy, (16,16), dist=('b',None), dtype='complex128')
A = distarray.fromfunction(funcy, (16,16), dist=('b',None), dtype='float128')

B = py_fftw.fft2(A)

C = py_fftw.ifft2(B)

print B

print C
