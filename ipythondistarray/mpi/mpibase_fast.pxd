include "../include/mpi.pxi"

cdef class Comm:
        
    cdef MPI_Comm comm
    cdef MPI_Group c_get_group(self)
    cdef int c_get_size(self)
    cdef int c_get_rank(self)
    cdef MPI_Comm c_clone(self)
    
cdef class Group:

    cdef MPI_Group group


