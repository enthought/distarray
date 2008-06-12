
cdef classs BaseDistArray:

    cdef int *dimensions
    cdef readonly int ndim
    cdef readonly object dtype
    cdef readonly int size
    cdef readonly int itemsize
    cdef readonly int nbytes
    cdef readonly object data
    
    cdef readonly object base_comm
    cdef readonly object comm
    cdef readonly int comm_size
    cdef readonly int comm_rank
    
    cdef readonly object dist
    cdef readonly int* distdims
    cdef int ndistdim
    cdef object map_classes
    cdef int *grid_shape
    cdef int *cart_coords
    cdef int *local_shape
    cdef object *maps
    cdef int local_size
    
    def __init__(self, shape, dtype=float, dist={0:'b'} , grid_shape=None,
                 comm=None, buf=None, offset=0):
        """Create a distributed memory array on a set of processors.
        """
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(dtype)
        self.size = reduce(lambda x,y: x*y, shape)
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size*self.itemsize
        self.data = None
        self.base = None
        self.ctypes = None
        
        # This order is extremely important and is shown by the arguments passed on to
        # subsequent _init_* methods.  It is critical that these _init_* methods are free
        # of side effects and stateless.  This means that they cannot set or get class or
        # instance attributes
        self.base_comm = init_base_comm(comm)
        self.comm_size = self.base_comm.Get_size()
        self.comm_rank = self.base_comm.Get_rank()
        
        self.dist = init_dist(dist, self.ndim)
        self.distdims = init_distdims(self.dist, self.ndim)
        self.ndistdim = len(self.distdims)
        self.map_classes = init_map_classes(self.dist)
        
        self.grid_shape = init_grid_shape(self.shape, grid_shape, 
            self.distdims, self.comm_size)
        self.comm = init_comm(self.base_comm, self.grid_shape, self.ndistdim)
        self.cart_coords = self.comm.Get_coords(self.comm_rank)
        self.local_shape, self.maps = init_local_shape_and_maps(self.shape, 
            self.grid_shape, self.distdims, self.map_classes)
        self.local_size = reduce(lambda x,y: x*y, self.local_shape)
        
        # At this point, everything is setup, but the memory has not been allocated.
        self._allocate(buf, offset)