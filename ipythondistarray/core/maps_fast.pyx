include "../include/python.pxi"

# cdef public class Map [ object MapObj, type MapType ]:
cdef class Map:

    def __init__(self, nglobal, nprocs):
        self.nglobal = nglobal
        self.nprocs = nprocs
        self.nlocal = self.nglobal/self.nprocs
        if self.nglobal%self.nprocs > 0:
            self.nlocal += 1
        print self.nglobal


cdef class BlockMap(Map):
        
    cdef int c_owner(BlockMap self, int global_index):
        cdef int o
        o = global_index/self.nlocal
        return o
    
    def owner(self, global_index):
        return self.c_owner(global_index)
        
    cdef int local_index(BlockMap self, int global_index):
        cdef int i
        i = global_index%self.nprocs
        return i
        
    cdef int global_index(BlockMap self, int owner, int local_index):
        cdef int i
        i = owner*self.nlocal + local_index
        return i

def test1():
    cdef BlockMap m
    m = BlockMap(16,4)
    for i from 0 <= i < 1000:
        m.owner(3)

def test2():
    cdef BlockMap m
    m = BlockMap(16,4)
    for i from 0 <= i < 1000:
        m.owner(3)

def test3():
    cdef BlockMap m
    m = BlockMap(16,4)
    for i from 0 <= i < 1000:
        m.c_owner(3)

cdef public foo(int i):
    return i


