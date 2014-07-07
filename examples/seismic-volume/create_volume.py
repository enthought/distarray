# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Create a fake 3D array that will pretend to be a seismic volume.

Usage:
    $ python create_volume.py --size NX NY NZ --file <filename> --key <keyname> --hdf5 --dnpy

The --size argument specifies the size of the 3D volume to create.
The --file argument specifies the output filename, default 'seismic.hdf5'.
The --key argument specifies the name of the data block in the HDF5 file, default 'seismic'.
The --hdf5 argument specifies to create an HDF5 file, default True.
The --dnpy argument specifies to create .dnpy files, default False.
Only one of --hdf5 or --dnpy can be specified.

The 3D float32 array is written as an HDF5 file, which can be processed by
the additional example load_volume.py. We try and make it look interesting.
The volume contains a couple of planes, which are angled. As ones passes
down past each plane, the data values show a Gaussian peak at the plane
position, and once past, the final value is larger than when started.
"""

from __future__ import print_function

import argparse
import os.path
import sys
from math import sqrt
from time import time

import numpy
import numpy.random
import h5py
from numpy import exp, float32

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray


# Physical size of volume.
PHYSICAL_X = 10.0
PHYSICAL_Y = 10.0
PHYSICAL_Z = 20.0

# Default array size of volume.
DEFAULT_ARRAY_SIZE = (256, 256, 1024)


def g(z, z0, A=100.0, B=40.0, C=2.5, mu=0.1):
    ''' A hopefully interesting looking function.

    Gaussian, with a step-like function past the gaussian peak.
    The peak is at depth z=z0.
    '''
    u = mu * (z - z0) ** 2
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
    nmag = sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nmag, ny / nmag, nz / nmag
    # Depth of peak.
    z0 = (D - nx * x - ny * y) / nz
    return z0


def create_horizon(shape, normal=(0.1, -0.2, 1.0), D=10.0):
    '''Get the horizon surface where we will place the peak.'''
    if len(shape) != 2:
        raise ValueError('Horizon shape must be 2D.')
    horizon = numpy.zeros(shape)
    for i in xrange(shape[0]):
        x = PHYSICAL_X * float(i) / float(shape[0])
        for j in xrange(shape[1]):
            y = PHYSICAL_Y * float(j) / float(shape[1])
            horizon[i, j] = p(x, y, normal=normal, D=D)
    return horizon


def create_horizons(shape):
    '''Create some horizons.'''
    horizon_1 = create_horizon(shape, normal=(0.1, -0.2, 1.0), D=10.0)
    params_1 = (100.0, 40.0, 0.0, 0.1)
    horizon_2 = create_horizon(shape, normal=(0.4, 0.1, 1.0), D=15.0)
    params_2 = (25.0, 15.0, 0.0, 0.7)
    rtn = [(horizon_1, params_1), (horizon_2, params_2)]
    return rtn


def create_volume(shape):
    ''' Create a fake seismic volume. We try to make it look interesting. '''
    if len(shape) != 3:
        raise ValueError('Volume shape must be 3D.')
    print('Creating volume array %d x %d x %d...' % (
        shape[0], shape[1], shape[2]))
    vol = numpy.zeros(shape, dtype=numpy.float32)
    print('Creating horizons...')
    horizons_shape = (shape[0], shape[1])
    horizons = create_horizons(horizons_shape)
    print('Filling array...')
    z = numpy.empty((shape[2],))
    for k in xrange(shape[2]):
        z[k] = PHYSICAL_Z * float(k) / float(shape[2])
    for i in xrange(shape[0]):
        print('Index', i + 1, 'of', shape[0], end='\r')
        sys.stdout.flush()
        for j in xrange(shape[1]):
            for horizon in horizons:
                horizon_plane, horizon_params = horizon
                z0 = horizon_plane[i, j]
                A, B, C, mu = horizon_params
                vol[i, j, :] += g(z, z0, A=A, B=B, C=C, mu=mu)
    print()
    # Add constant
    vol[:, :, :] += 2.5
    # Add randomness.
    rnd = numpy.random.randn(*shape)
    vol[:, :, :] += 2.0 * rnd
    print('Done.')
    return vol


def create_hdf5_file(volume, filename, key):
    '''Create an HDF5 file with the seismic volume.'''
    f = h5py.File(filename, 'w')
    dataset = f.create_dataset(key, volume.shape, dtype='f')
    dataset[...] = volume
    f.close()


def create_dnpy_files(volume, filename):
    ''' Create .dnpy files with the seismic volume. '''
    # Create context.
    context = Context()
    # Create a DistArray with the data.
    dist = ('b', 'b', 'n')
    array_shape = volume.shape
    distribution = Distribution(context, array_shape, dist=dist)
    da = DistArray(distribution, dtype=float32)
    # Fill the array.
    da[:, :, :] = volume[:, :, :]
    # Filename for save_dnpy() needs the full path,
    # and should strip any extension.
    filename = os.path.splitext(filename)[0]
    pathname = os.path.abspath(filename)
    # Write it.
    context.save_dnpy(pathname, da)


def main():
    # Parse arguments:
    #     --size NX NY NZ
    #     --file <filename>
    #     --key <keyname>
    #     --hdf5
    #     --dnpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',
                        nargs=3,
                        default=DEFAULT_ARRAY_SIZE,
                        type=int,
                        metavar=('NX', 'NY', 'NZ'),
                        help='Size (X, Y, Z) of seismic volume.')
    parser.add_argument('--file',
                        default='seismic.hdf5',
                        help='Name of output file.')
    parser.add_argument('--key',
                        default='seismic',
                        help='Name of HDF5 key for data.')
    parser.add_argument('--hdf5',
                        action='store_true',
                        help='Write output as an HDF5 file.')
    parser.add_argument('--dnpy',
                        action='store_true',
                        help='Write output as .dnpy files.')
    args = parser.parse_args()
    # Extract arguments and complain about invalid ones.
    shape = args.size
    filename = args.file
    key = args.key
    use_hdf5 = args.hdf5
    use_dnpy = args.dnpy
    # Pick either HDF5 or .dnpy
    if (use_hdf5 == False) and (use_dnpy == False):
        use_hdf5 = True
    if (use_hdf5 == True) and (use_dnpy == True):
        raise ValueError('Can only specify one of --hdf5 or --dnpy.')
    # Create the seismic volume and write it.
    t0 = time()
    vol = create_volume(shape)
    if use_hdf5:
        print('Creating hdf5 file...')
        create_hdf5_file(vol, filename=filename, key=key)
    elif use_dnpy:
        print('Creating dnpy files...')
        create_dnpy_files(vol, filename=filename)
    t1 = time()
    dt = t1 - t0
    print('Creation time: %.3f sec' % (dt))


if __name__ == '__main__':
    main()
