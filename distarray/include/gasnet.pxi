
cdef extern from "stdint.h":

    ctypedef unsigned int uintptr_t

cdef extern from "gasnet.h":

    # Return codes for gasnet API functions
    cdef enum:
        GASNET_OK 
        GASNET_ERR_RESOURCE 
        GASNET_ERR_BAD_ARG 
        GASNET_ERR_NOT_INIT
        GASNET_ERR_BARRIER_MISMATCH 
        GASNET_ERR_NOT_READY 

    # Constants
    cdef enum:
        GASNET_PAGESIZE

    # These functions convert between error codes and error names/desc
    char *gasnet_ErrorName(int errval)
    char *gasnet_ErrorDesc(int errval)

    # Basic types
    ctypedef unsigned int gasnet_node_t    # unsigned int
    ctypedef int gasnet_handle_t    # Opaque type
    ctypedef unsigned int gasnet_handler_t    # unsigned int
    ctypedef int gasnet_handlerarg_t  # 32 bit signed int
    ctypedef int gasnet_token_t    # Opaque type
    ctypedef unsigned int gasnet_register_value_t    # unsigned int
    ctypedef struct gasnet_handlerentry_t:
        gasnet_handler_t index
        void (*fnptr)()
    ctypedef struct gasnet_seginfo_t:
        void *addr
        uintptr_t size 
    
    # Function to init the gasnet library, like MPI_Init
    int gasnet_init(int *argc, char ***argv)
    int gasnet_attach(gasnet_handlerentry_t *table, int numentries,
    uintptr_t segsize, uintptr_t minheapoffset)
    uintptr_t gasnet_getMaxLocalSegmentSize()
    uintptr_t gasnet_getMaxGlobalSegmentSize()
    void gasnet_exit(int exitcode)
    
    # Job environment queries
    gasnet_node_t gasnet_mynode()
    gasnet_node_t gasnet_nodes()
    int gasnet_getSegmentInfo(gasnet_seginfo_t *seginfo_table, int numentries)
    char *gasnet_getenv(char *name)
