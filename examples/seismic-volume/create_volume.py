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

The 3D float32 array is written as either an HDF5 file, or a set of .dnpy files,
which can be processed by the additional example load_volume.py.
We try and make the volume look interesting.
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
from numpy import exp, linspace, float32

from distarray.dist import Context, Distribution
from distarray.dist.distarray import DistArray


# Physical size of volume.
PHYSICAL_X = 10.0
PHYSICAL_Y = 10.0
PHYSICAL_Z = 20.0

# Default array size of volume.
#DEFAULT_ARRAY_SIZE = (256, 256, 1024)
DEFAULT_ARRAY_SIZE = (32, 48, 256)


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


def scaled_points(num_points, total_size):
    ''' Get num_points values, ranging from 0.0 to total_size. '''
    p = linspace(0.0, total_size, num_points)
    return p


def create_horizon(context, shape, physical_x, physical_y, normal=(0.1, -0.2, 1.0), D=10.0):
    '''Get the horizon surface where we will place the peak.'''
    if len(shape) != 2:
        raise ValueError('Horizon shape must be 2D.')
    horizon = numpy.zeros(shape)
    for i in xrange(shape[0]):
        x = physical_x[i]
        for j in xrange(shape[1]):
            y = physical_y[j]
            horizon[i, j] = p(x, y, normal=normal, D=D)
    # Wrap it as distarray.
    horizon_dist = ('b', 'b')
    distribution = Distribution(context, shape, dist=horizon_dist)
    horizon_da = DistArray(distribution, dtype=float32)
    # Fill the array.
    horizon_da[:, :] = horizon[:, :]
    return horizon, horizon_da


def local_create_horizon(horizon_la, physical_x, physical_y, normal=(0.1, -0.2, 1.0), D=10.0):
    '''Get the horizon surface where we will place the peak.'''

    def p(x, y, normal=(0.1, -0.2, 1.0), D=10.0):
        '''Get the depth on the plane where we want to put the peak.'''
        from math import sqrt
        nx, ny, nz = normal
        nmag = sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nmag, ny / nmag, nz / nmag
        # Depth of peak.
        z0 = (D - nx * x - ny * y) / nz
        return z0

    def create_horizon(horizon, physical_x, physical_y, normal, D):
        #if len(horizon_da.shape) != 2:
        #    raise ValueError('Horizon shape must be 2D.')
        shape = horizon.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                x = physical_x[i, j]
                y = physical_y[i, j]
                horizon[i, j] = p(x, y, normal=normal, D=D)

    from distarray.local import LocalArray
    horizon = horizon_la.ndarray
    shape = horizon.shape
    # Get physical x and y values in an array the same shape as the horizon.
    px = numpy.empty(shape)
    py = numpy.empty(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            l_i, l_j = horizon_la.global_from_local((i, j))
            px[i, j] = physical_x[l_i]
            py[i, j] = physical_y[l_j]
    create_horizon(horizon, px, py, normal, D)
    res = LocalArray(horizon_la.distribution, buf=horizon)
    rtn = proxyize(res)
    return rtn



def distributed_create_horizon(context, shape, physical_x, physical_y, normal=(0.1, -0.2, 1.0), D=10.0):
    ''' Create a horizon. '''
    horizon_dist = ('b', 'b')
    distribution = Distribution(context, shape, dist=horizon_dist)
    horizon_da = DistArray(distribution, dtype=float32)
    context.apply(local_create_horizon, (horizon_da.key, physical_x, physical_y, normal, D))
    return horizon_da


def distributed_create_horizons(context, shape, physical_x, physical_y):
    '''Create some horizons.'''
    horizon_1_da = distributed_create_horizon(context, shape, physical_x, physical_y, normal=(0.1, -0.2, 1.0), D=10.0)
    params_1 = (100.0, 40.0, 0.0, 0.1)
    horizon_2_da = distributed_create_horizon(context, shape, physical_x, physical_y, normal=(0.4, 0.1, 1.0), D=15.0)
    params_2 = (25.0, 15.0, 0.0, 0.7)
    rtn = [(horizon_1_da, params_1), (horizon_2_da, params_2)]
    return rtn


def create_horizons(context, shape, physical_x, physical_y):
    '''Create some horizons.'''
    horizon_1, horizon_1_da = create_horizon(context, shape, physical_x, physical_y, normal=(0.1, -0.2, 1.0), D=10.0)
    params_1 = (100.0, 40.0, 0.0, 0.1)
    horizon_2, horizon_2_da = create_horizon(context, shape, physical_x, physical_y, normal=(0.4, 0.1, 1.0), D=15.0)
    params_2 = (25.0, 15.0, 0.0, 0.7)
    rtn = [(horizon_1, horizon_1_da, params_1), (horizon_2, horizon_2_da, params_2)]
    return rtn


def create_volume(context, shape):
    ''' Create a fake seismic volume. We try to make it look interesting. '''
    if len(shape) != 3:
        raise ValueError('Volume shape must be 3D.')
    print('Creating volume array %d x %d x %d...' % (
        shape[0], shape[1], shape[2]))
    vol = numpy.zeros(shape, dtype=numpy.float32)
    print('Physical coords...')
    x = scaled_points(shape[0], PHYSICAL_X)
    y = scaled_points(shape[1], PHYSICAL_Y)
    z = scaled_points(shape[2], PHYSICAL_Z)
    print('Creating horizons...')
    horizons_shape = (shape[0], shape[1])
    horizons = create_horizons(context, horizons_shape, x, y)
    print('Filling array...')
    for i in xrange(shape[0]):
        print('Index', i + 1, 'of', shape[0], end='\r')
        sys.stdout.flush()
        for j in xrange(shape[1]):
            for horizon in horizons:
                horizon_plane, horizon_plane_da, horizon_params = horizon
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


def local_add_horizon(la, lh, z, h_params):
    ''' Filter a local array via 3-point average over z axis. '''

    def g(z, z0, A=100.0, B=40.0, C=2.5, mu=0.1):
        ''' A hopefully interesting looking function.

        Gaussian, with a step-like function past the gaussian peak.
        The peak is at depth z=z0.
        '''
        from numpy import exp
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

    def add_horizon(a, h, z, h_params):
        ''' Add the effect of a horizon to the seismic volume data. '''
        shape = h.shape
        A, B, C, mu = h_params
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                z0 = h[i, j]
                a[i, j, :] += g(z, z0, A=A, B=B, C=C, mu=mu)
        return a

    a = la.ndarray
    h = lh.ndarray
    add_horizon(a, h, z, h_params)
    return la


def local_add_random(la):
    ''' Add randomness and a constant to the local array data. '''

    def add_random(a):
        ''' Add randomness and a constant to the volume. '''
        # Add constant
        a[:, :, :] += 2.5
        # Add randomness.
        shape = a.shape
        rnd = numpy.random.randn(*shape)
        a[:, :, :] += 2.0 * rnd

    a = la.ndarray
    add_random(a)
    return la


def distributed_create_volume(context, shape):
    ''' Create a fake seismic volume. We try to make it look interesting. '''
    if len(shape) != 3:
        raise ValueError('Volume shape must be 3D.')
    print('Creating volume array %d x %d x %d...' % (
        shape[0], shape[1], shape[2]))
    dist = ('b', 'b', 'n')
    distribution = Distribution(context, shape, dist=dist)
    da = context.zeros(distribution, dtype=float32)
    print('Physical coords...')
    x = scaled_points(shape[0], PHYSICAL_X)
    y = scaled_points(shape[1], PHYSICAL_Y)
    z = scaled_points(shape[2], PHYSICAL_Z)
    print('Creating horizons...')
    horizons_shape = (shape[0], shape[1])
    horizons = create_horizons(context, horizons_shape, x, y)
    print('Distributed...')
    dist_horizons = distributed_create_horizons(context, horizons_shape, x, y)
    print('Filling array...')
    # Add data for each horizon...
    #for ih, horizon in enumerate(horizons):
    for ih, horizon in enumerate(dist_horizons):
        print('Horizon %d...' % (ih + 1))
        #horizon_np, horizon_da, horizon_params = horizon
        horizon_da, horizon_params = horizon
        context.apply(local_add_horizon, (da.key, horizon_da.key, z, horizon_params))
    # Add global randomness.
    print('Adding randomness...')
    context.apply(local_add_random, (da.key,))
    print('Done.')
    return da


def create_hdf5_file(da_volume, filename, key):
    '''Create an HDF5 file with the seismic volume.'''
    f = h5py.File(filename, 'w')
    dataset = f.create_dataset(key, da_volume.shape, dtype='f')
    dataset[...] = da_volume.tondarray()
    f.close()


def create_dnpy_files(da_volume, filename):
    ''' Create .dnpy files with the seismic volume. '''
    # Filename for save_dnpy() needs the full path,
    # and should strip any extension.
    filename = os.path.splitext(filename)[0]
    pathname = os.path.abspath(filename)
    # Write it.
    da_volume.context.save_dnpy(pathname, da_volume)


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
    # Create context.
    context = Context()
    # Create the seismic volume and write it.
    t0 = time()
    vol = create_volume(context, shape)
    if False:
        # Wrap as DistArray.
        dist = ('b', 'b', 'n')
        distribution = Distribution(context, shape, dist=dist)
        da_vol = context.empty(distribution, dtype=float32)
        da_vol[:, :, :] = vol[:, :, :]
    else:
        da_vol = distributed_create_volume(context, shape)
    if use_hdf5:
        print('Creating hdf5 file...')
        create_hdf5_file(da_vol, filename=filename, key=key)
    elif use_dnpy:
        print('Creating dnpy files...')
        create_dnpy_files(da_vol, filename=filename)
    t1 = time()
    dt = t1 - t0
    print('Creation time: %.3f sec' % (dt))


if __name__ == '__main__':
    main()
