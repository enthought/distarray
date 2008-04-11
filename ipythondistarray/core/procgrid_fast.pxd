
cdef class ProcGrid:

    cdef readonly int ndim
    cdef int *strides
    cdef public object shape
    
    cdef int c_global_rank(self, int *proc_indices)

