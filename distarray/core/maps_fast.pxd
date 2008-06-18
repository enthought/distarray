cdef class Map:
    cdef public int shape
    cdef public int grid_shape
    cdef public int local_shape
    cdef int owner_c(self, int i)
    cdef int local_index_c(self, int i)
    cdef int global_index_c(self, int owner, int p)

cdef class BlockMap(Map):
    pass

cdef class CyclicMap(Map):
    pass

cdef class BlockCyclicMap(Map):
    pass


