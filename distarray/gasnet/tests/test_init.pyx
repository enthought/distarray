# mpicc -o testmpi.o -I/Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 testmpi.c
# mpicc -dynamiclib -o testmpi.so testmpi.o -framework Python

# gcc -o test_init.o \
# -I/Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 \
# -I/usr/local/gasnet/include -I/usr/local/gasnet/mpi-conduit -L/usr/local/gasnet/lib -DGASNET_PAR test_init.c -lammpi -lgasnet-mpi-par -framework Python

# mpicc -I/Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 -I/usr/local/gasnet/include -I/usr/local/gasnet/include/mpi-conduit -DGASNET_PAR -c test_init.c -o test_init.o
# mpicc -dynamiclib -o test_init.so test_init.o -L/usr/local/gasnet/lib -framework Python -lammpi -lgasnet-mpi-par -lgasnet_tools-par

include '../../include/gasnet.pxi'


import sys

def main():
    cdef int argc = 0
    cdef char **argv = NULL
    cdef gasnet_node_t rank
    cdef gasnet_node_t size
            
    gasnet_init(&argc,&argv)
    
    # rank = gasnet_mynode()
    # size = gasnet_nodes()
    # print rank, size
    
    gasnet_exit(0)



