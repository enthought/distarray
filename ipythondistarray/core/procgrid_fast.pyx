
include "../include/python.pxi"
include "../include/stdlib.pxi"

cdef class ProcGrid:

    def __cinit__(self, object shape):
        cdef Py_ssize_t i
    
        if PySequence_Check(shape):
            self.ndim = PySequence_Length(shape)
            self.strides = <int *>PyMem_Malloc(self.ndim*sizeof(int))
            for i from 0 <= i < self.ndim:
                self.strides[i] = shape[self.ndim-i-1]
            self.strides[self.ndim-1] = 1
            self.shape = tuple(shape)
        else:
            raise TypeError("The shape argument must be a sequence")

    def __dealloc__(self):
        PyMem_Free(self.strides)
    
    cdef int c_global_rank(self, int *proc_indices):
        cdef Py_ssize_t i
        cdef int rank = 0
        
        for i from 0 <= i < self.ndim:
            rank += self.strides[i]*proc_indices[i]
        return rank
    
    def __getitem__(self, object indices):
  
        cdef Py_ssize_t i
        cdef int *proc_indices
        cdef rank
        
        if PyTuple_Check(indices):
            if PyTuple_Size(indices) != self.ndim:
                raise IndexError("Number of indices must match ndim (%s)" % self.ndim)

            proc_indices = <int *>malloc(self.ndim*sizeof(int))
            for i from 0 <= i < self.ndim:
                proc_indices[i] = <object>PyTuple_GET_ITEM(indices, i)
            rank = self.c_global_rank(proc_indices)
            free(proc_indices)
            return rank
        else:
            raise IndexError("This class only supports simple indexing")


