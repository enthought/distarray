# include "../include/python.pxi"

import inspect

from ipythondistarray.core.error import InvalidMapCode

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

class MapRegistry(object):
    
    def __init__(self):
        self.maps = {}
        
    def register_map(self, code, m):
        if inspect.isclass(m):
            if issubclass(m, Map):
                self.maps[code] = m
            else:
                raise TypeError("Must register a Map subclass.")
        else:
            raise TypeError("Must register a class")

    def get_map_class(self, code):
        m = self.maps.get(code)
        if m is None:
            if inspect.isclass(code): 
                if issubclass(code, Map):
                    return code
                else:
                    raise InvalidMapCode("Not a Map subclass or a valid map code: %s"%code)
            else:
                raise InvalidMapCode("Not a Map subclass or a valid map code: %s"%code)
        else:
            return m
            
            
_map_registry = MapRegistry()
register_map = _map_registry.register_map
get_map_class = _map_registry.get_map_class

register_map('b', BlockMap)
register_map('c', CyclicMap)
register_map('bc', BlockCyclicMap)

