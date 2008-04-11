# mpicc -o testmpi.o -I/Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 testmpi.c
# mpicc -dynamiclib -o testmpi.so testmpi.o -framework Python

include '../../include/mpi.pxi'


import sys

def main():
    cdef int argc = 0
    cdef char **argv = NULL
    cdef int rank
    cdef int size
    cdef int i
            
    MPI_Init(&argc,&argv)

    MPI_Comm_size(MPI_COMM_WORLD, &size)

    MPI_Comm_rank(MPI_COMM_WORLD, &rank)

    print rank, size
    MPI_Finalize()

