"""
Create a fake 3D array that will pretend to be a seismic volume.
"""

from math import exp
import numpy, numpy.random
import h5py

#SIZE = (10, 10, 20)
SIZE = (4, 4, 10)

def g(z, z0):
    '''Gaussian.'''
    A = 100.0
    C = 2.5
    mu = 2.0
    g = A * exp(-mu * (z - z0)**2) + C
    return g

def p(x, y):
    '''Get the depth on the plane.'''
    nx, ny, nz = 0.1, -0.2, 1.0
    D = 0.5 * SIZE[2]
    # Depth of peak.
    z0 = (D - nx * x - ny * y) / nz
    return z0

def create_plane():
    '''Get the plane for the peak.'''
    horizon = numpy.zeros((SIZE[0], SIZE[1]))
    for i in xrange(SIZE[0]):
        for j in xrange(SIZE[1]):
            horizon[i, j] = p(i, j)
    return horizon

def create_volume():
    #vol = numpy.random.randn(SIZE[0], SIZE[1], SIZE[2])
    vol = numpy.zeros(SIZE, dtype=numpy.float32)
    horizon = create_plane()
    for i in xrange(SIZE[0]):
        for j in xrange(SIZE[1]):
            z0 = horizon[i, j]
            for k in xrange(SIZE[2]):
                vol[i, j, k] = g(k, z0)
    return vol

def create_file(volume):
    '''Create an HDF5 file with the seismic volume.'''
    f = h5py.File("seismic.hdf5", "w")
    dataset = f.create_dataset("seismic", SIZE, dtype='f')
    print "Dataset dataspace is", dataset.shape
    print "Dataset Numpy datatype is", dataset.dtype
    print "Dataset name is", dataset.name
    print "Dataset is a member of the group", dataset.parent
    print "Dataset was created in the file", dataset.file
    dataset[...] = volume
    f.close()
    
def main():
    vol = create_volume()
    print vol
    create_file(vol)

if __name__ == '__main__':
    main()

