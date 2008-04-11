cdef public class Map [ object MapObj, type MapType ]:
    
    cdef int nglobal
    cdef int nprocs
    cdef int nlocal
    
cdef public class BlockMap(Map) [ object BlockMapObj, type BlockMapType ]:
        
    cdef int c_owner(BlockMap self, int global_index)      
    cdef int local_index(BlockMap self, int global_index)  
    cdef int global_index(BlockMap self, int owner, int local_index)

