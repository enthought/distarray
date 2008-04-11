
include 'mpi.pxi'

cdef extern from *:

    cdef enum:
        MPI_ERR_ACCESS
        MPI_ERR_AMODE
        MPI_ERR_ASSERT
        MPI_ERR_BAD_FILE
        MPI_ERR_BASE
        MPI_ERR_CONVERSION
        MPI_ERR_DISP
        MPI_ERR_DUP_DATAREP
        MPI_ERR_FILE_EXISTS
        MPI_ERR_FILE_IN_USE
        MPI_ERR_FILE
        MPI_ERR_INFO_KEY
        MPI_ERR_INFO_NOKEY
        MPI_ERR_INFO_VALUE
        MPI_ERR_INFO
        MPI_ERR_IO
        MPI_ERR_KEYVAL
        MPI_ERR_LOCKTYPE
        MPI_ERR_NAME
        MPI_ERR_NO_MEM
        MPI_ERR_NOT_SAME
        MPI_ERR_NO_SPACE
        MPI_ERR_NO_SUCH_FILE
        MPI_ERR_PORT
        MPI_ERR_QUOTA
        MPI_ERR_READ_ONLY
        MPI_ERR_RMA_CONFLICT
        MPI_ERR_RMA_SYNC
        MPI_ERR_SERVICE
        MPI_ERR_SIZE
        MPI_ERR_SPAWN
        MPI_ERR_UNSUPPORTED_DATAREP
        MPI_ERR_UNSUPPORTED_OPERATION
        MPI_ERR_WIN

    cdef enum:
        MPI_IN_PLACE
        MPI_LOCK_EXCLUSIVE
        MPI_LOCK_SHARED
        MPI_ROOT
    
    cdef enum:
        MPI_ADDRESS_KIND
        MPI_INTEGER_KIND
        MPI_OFFSET_KIND
        
    cdef enum:
        MPI_MAX_DATAREP_STRING
        MPI_MAX_INFO_KEY
        MPI_MAX_INFO_VAL
        MPI_MAX_OBJECT_NAME
        MPI_MAX_PORT_NAME
        
    cdef enum:
        MPI_WCHAR
    
    cdef enum:
        MPI_Fint
    
    cdef enum:
        MPI_UNSIGNED_LONG_LONG
        MPI_SIGNED_CHAR
    
    cdef enum:
        MPI_APPNUM
        MPI_LASTUSEDCODE
        MPI_UNIVERSE_SIZE
        MPI_WIN_BASE
        MPI_WIN_DISP_UNIT
        MPI_WIN_SIZE

    cdef enum:    
        MPI_REPLACE

    cdef enum:
        MPI_FILE_NULL
        MPI_INFO_NULL
        MPI_WIN_NULL
    
    cdef enum:
        MPI_MODE_APPEND
        MPI_MODE_CREATE
        MPI_MODE_DELETE_ON_CLOSE
        MPI_MODE_EXCL
        MPI_MODE_NOCHECK
        MPI_MODE_NOPRECEDE
        MPI_MODE_NOPUT
        MPI_MODE_NOSTORE
        MPI_MODE_NOSUCCEED
        MPI_MODE_RDONLY
        MPI_MODE_RDWR
        MPI_MODE_SEQUENTIAL
        MPI_MODE_UNIQUE_OPEN
        MPI_MODE_WRONLY
    
    cdef enum:
        MPI_COMBINER_CONTIGUOUS
        MPI_COMBINER_DARRAY
        MPI_COMBINER_DUP
        MPI_COMBINER_F90_COMPLEX
        MPI_COMBINER_F90_INTEGER
        MPI_COMBINER_F90_REAL
        MPI_COMBINER_HINDEXED_INTEGER
        MPI_COMBINER_HINDEXED
        MPI_COMBINER_HVECTOR_INTEGER
        MPI_COMBINER_HVECTOR
        MPI_COMBINER_INDEXED_BLOCK
        MPI_COMBINER_INDEXED
        MPI_COMBINER_NAMED
        MPI_COMBINER_RESIZED
        MPI_COMBINER_STRUCT_INTEGER
        MPI_COMBINER_STRUCT
        MPI_COMBINER_SUBARRAY
        MPI_COMBINER_VECTOR
    
    cdef enum:
        MPI_THREAD_FUNNELED
        MPI_THREAD_MULTIPLE
        MPI_THREAD_SERIALIZED
        MPI_THREAD_SINGLE
    
    cdef enum:
        MPI_DISPLACEMENT_CURRENT
        MPI_DISTRIBUTE_BLOCK
        MPI_DISTRIBUTE_CYCLIC
        MPI_DISTRIBUTE_DFLT_DARG
        MPI_DISTRIBUTE_NONE
        MPI_ORDER_C
        MPI_ORDER_FORTRAN
        MPI_SEEK_CUR
        MPI_SEEK_END
        MPI_SEEK_SET
    
    ctypedef int MPI_File
    ctypedef int MPI_Info
    ctyepdef int MPI_Win
    
    ctyepdef int MPI_ARGVS_NULL
    ctyepdef int MPI_ARGV_NULL
    ctyepdef int MPI_ERRCODES_IGNORE
    ctyepdef int MPI_STATUSES_IGNORE
    ctyepdef int MPI_STATUS_IGNORE
    
    ctyepdef int MPI_F_STATUSES_IGNORE
    ctyepdef int MPI_F_STATUS_IGNORE
    
    ctyepdef int MPI_SUBVERSION
    ctyepdef int MPI_VERSION
    
    int MPI_Get_version(int *version, int *subversion) 

    int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr) 
    MPI_Fint MPI_Comm_c2f(MPI_Comm comm) 
    int MPI_Comm_create_errhandler(MPI_Comm_errhandler_fn *function, MPI_Errhandler *errhandler) 
    MPI_Comm MPI_Comm_f2c(MPI_Fint comm) 
    int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler) 
    int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) 
    MPI_Fint MPI_File_c2f(MPI_File file) 
    int MPI_File_create_errhandler(MPI_File_errhandler_fn *function, MPI_Errhandler *errhandler) 
    MPI_File MPI_File_f2c(MPI_Fint file) 
    int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler) 
    int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler) 
    int MPI_Finalized(int *flag) 
    int MPI_Free_mem(void *base) 
    int MPI_Get_address(void *location, MPI_Aint *address) 
    MPI_Fint MPI_Group_c2f(MPI_Group group) 
    MPI_Group MPI_Group_f2c(MPI_Fint group) 
    MPI_Fint MPI_Info_c2f(MPI_Info info) 
    int MPI_Info_create(MPI_Info *info) 
    int MPI_Info_delete(MPI_Info info, char *key) 
    int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo) 
    MPI_Info MPI_Info_f2c(MPI_Fint info) 
    int MPI_Info_free(MPI_Info *info) 
    int MPI_Info_get(MPI_Info info, char *key, int valuelen, char *value, int *flag) 
    int MPI_Info_get_nkeys(MPI_Info info, int *nkeys) 
    int MPI_Info_get_nthkey(MPI_Info info, int n, char *key) 
    int MPI_Info_get_valuelen(MPI_Info info, char *key, int *valuelen, int *flag) 
    int MPI_Info_set(MPI_Info info, char *key, char *value) 
    MPI_Fint MPI_Op_c2f(MPI_Op op) 
    MPI_Op MPI_Op_f2c(MPI_Fint op) 
    int MPI_Pack_external(char *datarep, void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position) 
    int MPI_Pack_external_size(char *datarep, int incount, MPI_Datatype datatype, MPI_Aint *size) 
    MPI_Fint MPI_Request_c2f(MPI_Request request) 
    MPI_Request MPI_Request_f2c(MPI_Fint request) 
    int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status) 
    int MPI_Status_c2f(MPI_Status *c_status, MPI_Fint *f_status) 
    int MPI_Status_f2c(MPI_Fint *f_status, MPI_Status *c_status) 
    MPI_Fint MPI_Type_c2f(MPI_Datatype datatype) 
    int MPI_Type_create_darray(int size, int rank, int ndims, int array_of_gsizes[], int array_of_distribs[], int array_of_dargs[], int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype) 
    int MPI_Type_create_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype) 
    int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype) 
    int MPI_Type_create_indexed_block(int count, int blocklength, int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype) 
    int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype) 
    int MPI_Type_create_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[], MPI_Datatype array_of_types[], MPI_Datatype *newtype) 
    int MPI_Type_create_subarray(int ndims, int array_of_sizes[], int array_of_subsizes[], int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype) 
    MPI_Datatype MPI_Type_f2c(MPI_Fint datatype) 
    int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent) 
    int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent) 
    int MPI_Unpack_external(char *datarep, void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype) 
    MPI_Fint MPI_Win_c2f(MPI_Win win) 
    int MPI_Win_create_errhandler(MPI_Win_errhandler_fn *function, MPI_Errhandler *errhandler) 
    MPI_Win MPI_Win_f2c(MPI_Fint win) 
    int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler) 
    int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler) 

    int MPI_Close_port(char *port_name) 
    int MPI_Comm_accept(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) 
    int MPI_Comm_connect(char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) 
    int MPI_Comm_disconnect(MPI_Comm *comm) 
    int MPI_Comm_get_parent(MPI_Comm *parent) 
    int MPI_Comm_join(int fd, MPI_Comm *intercomm) 
    int MPI_Comm_spawn(char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]) 
    int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[], int array_of_maxprocs[], MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]) 
    int MPI_Lookup_name(char *service_name, MPI_Info info, char *port_name) 
    int MPI_Open_port(MPI_Info info, char *port_name) 
    int MPI_Publish_name(char *service_name, MPI_Info info, char *port_name) 
    int MPI_Unpublish_name(char *service_name, MPI_Info info, char *port_name) 

    int MPI_Accumulate(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) 
    int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) 
    int MPI_Put(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win) 
    int MPI_Win_complete(MPI_Win win) 
    int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win) 
    int MPI_Win_fence(int assert, MPI_Win win) 
    int MPI_Win_free(MPI_Win *win) 
    int MPI_Win_get_group(MPI_Win win, MPI_Group *group) 
    int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win) 
    int MPI_Win_post(MPI_Group group, int assert, MPI_Win win) 
    int MPI_Win_start(MPI_Group group, int assert, MPI_Win win) 
    int MPI_Win_test(MPI_Win win, int *flag) 
    int MPI_Win_unlock(int rank, MPI_Win win) 
    int MPI_Win_wait(MPI_Win win) 

    int MPI_Alltoallw(void *sendbuf, int sendcounts[], int sdispls[], MPI_Datatype sendtypes[], void *recvbuf, int recvcounts[], int rdispls[], MPI_Datatype recvtypes[], MPI_Comm comm) 
    int MPI_Exscan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) 

    int MPI_Add_error_class(int *errorclass) 
    int MPI_Add_error_code(int errorclass, int *errorcode) 
    int MPI_Add_error_string(int errorcode, char *string) 
    int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode) 
    int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn, MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval, void *extra_state) 
    int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval) 
    int MPI_Comm_free_keyval(int *comm_keyval) 
    int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag) 
    int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen) 
    int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val) 
    int MPI_Comm_set_name(MPI_Comm comm, char *comm_name) 
    int MPI_File_call_errhandler(MPI_File fh, int errorcode) 
    int MPI_Grequest_complete(MPI_Request request) 
    int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request) 
    int MPI_Init_thread(int *argc, char *((*argv)[]), int required, int *provided) 
    int MPI_Is_thread_main(int *flag) 
    int MPI_Query_thread(int *provided) 
    int MPI_Status_set_cancelled(MPI_Status *status, int flag) 
    int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count) 
    int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn, MPI_Type_delete_attr_function *type_delete_attr_fn, int *type_keyval, void *extra_state) 
    int MPI_Type_delete_attr(MPI_Datatype type, int type_keyval) 
    int MPI_Type_dup(MPI_Datatype type, MPI_Datatype *newtype) 
    int MPI_Type_free_keyval(int *type_keyval) 
    int MPI_Type_get_attr(MPI_Datatype type, int type_keyval, void *attribute_val, int *flag) 
    int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]) 
    int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) 
    int MPI_Type_get_name(MPI_Datatype type, char *type_name, int *resultlen) 
    int MPI_Type_set_attr(MPI_Datatype type, int type_keyval, void *attribute_val) 
    int MPI_Type_set_name(MPI_Datatype type, char *type_name) 
    int MPI_Win_call_errhandler(MPI_Win win, int errorcode) 
    int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn, MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval, void *extra_state) 
    int MPI_Win_delete_attr(MPI_Win win, int win_keyval) 
    int MPI_Win_free_keyval(int *win_keyval) 
    int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag) 
    int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen) 
    int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val) 
    int MPI_Win_set_name(MPI_Win win, char *win_name) 

    int MPI_File_close(MPI_File *fh) 
    int MPI_File_delete(char *filename, MPI_Info info) 
    int MPI_File_get_amode(MPI_File fh, int *amode) 
    int MPI_File_get_atomicity(MPI_File fh, int *flag) 
    int MPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset *disp) 
    int MPI_File_get_group(MPI_File fh, MPI_Group *group) 
    int MPI_File_get_info(MPI_File fh, MPI_Info *info_used) 
    int MPI_File_get_position(MPI_File fh, MPI_Offset *offset) 
    int MPI_File_get_position_shared(MPI_File fh, MPI_Offset *offset) 
    int MPI_File_get_size(MPI_File fh, MPI_Offset *size) 
    int MPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent) 
    int MPI_File_get_view(MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype, char *datarep) 
    int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_iread_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_iread_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_iwrite(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_iwrite_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) 
    int MPI_File_open(MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *fh) 
    int MPI_File_preallocate(MPI_File fh, MPI_Offset size) 
    int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_read_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_read_all_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence) 
    int MPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence) 
    int MPI_File_set_atomicity(MPI_File fh, int flag) 
    int MPI_File_set_info(MPI_File fh, MPI_Info info) 
    int MPI_File_set_size(MPI_File fh, MPI_Offset size) 
    int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, char *datarep, MPI_Info info) 
    int MPI_File_sync(MPI_File fh) 
    int MPI_File_write(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_write_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_write_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_write_all_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_write_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_write_at_all_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_write_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status)
    int MPI_File_write_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) 
    int MPI_File_write_ordered_end(MPI_File fh, void *buf, MPI_Status *status) 
    int MPI_File_write_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) 
    int MPI_Register_datarep(char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state) 

    int MPI_Type_create_f90_complex(int p, int r, MPI_Datatype *newtype) 
    int MPI_Type_create_f90_integer(int r, MPI_Datatype *newtype) 
    int MPI_Type_create_f90_real(int p, int r, MPI_Datatype *newtype) 
    int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *type) 

    ctypedef int MPI_Comm_copy_attr_function(MPI_Comm oldcomm, int comm_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag)
    ctypedef int MPI_Comm_delete_attr_function(MPI_Comm comm, int comm_keyval, void *attribute_val, void *extra_state)
    ctypedef void MPI_Comm_errhandler_fn(MPI_Comm *, int *, ...)
    ctypedef int MPI_Datarep_conversion_function(void *userbuf, MPI_Datatype datatype, int count, void *filebuf, MPI_Offset position, void *extra_state)
    ctypedef int MPI_Datarep_extent_function(MPI_Datatype datatype, MPI_Aint *file_extent, void *extra_state)
    ctypedef void MPI_File_errhandler_fn(MPI_File *, int *, ...)
    ctypedef int MPI_Grequest_cancel_function(void *extra_state, int complete)
    ctypedef int MPI_Grequest_free_function(void *extra_state)
    ctypedef int MPI_Grequest_query_function(void *extra_state, MPI_Status *status)
    ctypedef int MPI_Type_copy_attr_function(MPI_Datatype oldtype, int type_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag)
    ctypedef int MPI_Type_delete_attr_function(MPI_Datatype type, int type_keyval, void *attribute_val, void *extra_state)
    ctypedef int MPI_Win_copy_attr_function(MPI_Win oldwin, int win_keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag)
    ctypedef int MPI_Win_delete_attr_function(MPI_Win win, int win_keyval, void *attribute_val, void *extra_state)
    ctypedef void MPI_Win_errhandler_fn(MPI_Win *, int *, ...)

    
    
    
    
        
