"""
Create a fake 3D array that will pretend to be a seismic volume.

We try and make it look interesting.
"""

from math import sqrt
from numpy import exp
import numpy, numpy.random
import h5py

# Physical size of volume.
PHYSICAL_SIZE = (10.0, 10.0, 20.0)

# Array size of volume.
#ARRAY_SIZE = (2, 2, 10)
#ARRAY_SIZE = (4, 4, 25)
#ARRAY_SIZE = (32, 32, 256)
ARRAY_SIZE = (512, 512, 1024)
#ARRAY_SIZE = (512, 512, 2048)


def g(z, z0, A=100.0, B=40.0, C=2.5, mu=0.1):
    ''' A hopefully interesting looking function.

    Gaussian, with a step-like function past the gaussian peak.
    The peak is at depth z=z0.
    '''
    u = mu * (z - z0)**2
    v = mu * (z - z0)
    # Gaussian.
    g = A * exp(-u) + C
    # Smooth step.
    exp_p = exp(v)
    exp_m = exp(-v)
    h = B * (exp_p - exp_m) / (exp_p + exp_m)
    r = g + h
    return r

def p(x, y, normal=(0.1, -0.2, 1.0), D=10.0):
    '''Get the depth on the plane where we want to put the peak.'''
    nx, ny, nz = normal
    nmag = sqrt(nx*nx + ny*ny + nz*nz)
    nx, ny, nz = nx / nmag, ny / nmag, nz / nmag
    # Depth of peak.
    z0 = (D - nx * x - ny * y) / nz
    return z0

def get_physical(index, dimension):
    ''' Convert array index into physical value.'''
    return PHYSICAL_SIZE[dimension] * float(index) / float(ARRAY_SIZE[dimension])

def get_x(i):
    return get_physical(i, 0)

def get_y(j):
    return get_physical(j, 1)

def get_z(k):
    return get_physical(k, 2)

def create_horizon(normal=(0.1, -0.2, 1.0), D=10.0):
    '''Get the horizon surface where we will place the peak.'''
    horizon = numpy.zeros((ARRAY_SIZE[0], ARRAY_SIZE[1]))
    for i in xrange(ARRAY_SIZE[0]):
        x = get_x(i)
        for j in xrange(ARRAY_SIZE[1]):
            y = get_y(j)
            horizon[i, j] = p(x, y, normal=normal, D=D)
    return horizon

def create_horizons():
    '''Create some horizons.'''
    horizon_1 = create_horizon(normal=(0.1, -0.2, 1.0), D=10.0)
    params_1 = (100.0, 40.0, 0.0, 0.1)
    horizon_2 = create_horizon(normal=(0.4, 0.1, 1.0), D=15.0)
    params_2 = (25.0, 15.0, 0.0, 0.7)
    rtn = [(horizon_1, params_1), (horizon_2, params_2)]
    return rtn

def create_volume():
    ''' Create a fake seismic volume. We try to make it look interesting. '''
    print 'Creating volume array...'
    vol = numpy.zeros(ARRAY_SIZE, dtype=numpy.float32)
    print 'Creating horizons...'
    horizons = create_horizons()
    print 'Filling array...'
    z = numpy.empty((ARRAY_SIZE[2],))
    for k in xrange(ARRAY_SIZE[2]):
        z[k] = get_z(k)
    for i in xrange(ARRAY_SIZE[0]):
        print 'Index', i, 'of', ARRAY_SIZE[0]
        for j in xrange(ARRAY_SIZE[1]):
            for horizon in horizons:
                horizon_plane, horizon_params = horizon
                z0 = horizon_plane[i, j]
                A, B, C, mu = horizon_params
                vol[i, j, :] += g(z, z0, A=A, B=B, C=C, mu=mu)
    # Add constant
    vol[:, :, :] += 2.5
    # Add randomness.
    rnd = numpy.random.randn(*ARRAY_SIZE)
    vol[:, :, :] += 2.0 * rnd
    print 'Done.'
    return vol

def create_file(volume, filename, key):
    '''Create an HDF5 file with the seismic volume.'''
    f = h5py.File(filename, 'w')
    dataset = f.create_dataset(key, ARRAY_SIZE, dtype='f')
    dataset[...] = volume
    print "Dataset dataspace is", dataset.shape
    print "Dataset Numpy datatype is", dataset.dtype
    print "Dataset name is", dataset.name
    print "Dataset is a member of the group", dataset.parent
    print "Dataset was created in the file", dataset.file
    f.close()

def main():
    filename = 'seismic.hdf5'
    key = 'seismic'
    vol = create_volume()
    if False:
        print vol
    create_file(vol, filename=filename, key=key)

if __name__ == '__main__':
    main()
