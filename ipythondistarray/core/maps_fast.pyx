include "../include/python.pxi"

cdef class Map:

    def __init__(self, shape, grid_shape):
        self.shape = shape
        self.grid_shape = grid_shape
        self.local_shape = self.shape/self.grid_shape
        if self.shape%self.grid_shape > 0:
            self.local_shape += 1
    
    cdef int owner_c(self, int i):
        raise NotImplementedError("this method needs to be implemented in a subclass")
    
    cdef int local_index_c(self, int i):
        raise NotImplementedError("this method needs to be implemented in a subclass")
    
    cdef int global_index_c(self, int owner, int p):
        raise NotImplementedError("this method needs to be implemented in a subclass")
    
    def owner(self, i):
        return self.owner_c(i)
    
    def local_index(self, i):
        return self.local_index_c(i)
    
    def global_index(self, owner, p):
        return self.global_index_c(owner, p)


cdef class BlockMap(Map):
    
    cdef int owner_c(self, int i):
        return i/self.local_shape
    
    cdef int local_index_c(self, int i):
        return i%self.local_shape
    
    cdef int global_index_c(self, int owner, int p):
        return owner*self.local_shape + p


cdef class CyclicMap(Map):
    
    cdef int owner_c(self, int i):
        return i%self.grid_shape
    
    cdef int local_index_c(self, int i):
        return i/self.grid_shape

    cdef int global_index_c(self, int owner, int p):
        return owner + p*self.grid_shape

cdef class BlockCyclicMap(Map):
    pass



