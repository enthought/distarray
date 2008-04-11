include '../include/mpi.pxi'

# Importing this calls MPI_Init and insures that MPI_Finalize is called
# at the right time.
from mpi4py import MPI


cdef class Comm:
            
    def __init__(self, comm=None):
        if comm is None:
            self.comm
            
    cdef MPI_Group c_get_group(self):
        pass
    
    cdef int c_get_size(self):
        cdef int size, ierr
        ierr = MPI_Comm_size(self.comm, &size)
        return size
        
    cdef int c_get_rank(self):
        cdef int rank, ierr
        ierr = MPI_Comm_rank(self.comm, &rank)
        return rank
        
    cdef MPI_Comm c_clone(self):
        pass
        
cdef class Group:

    pass