# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

import numpy as np
from distarray.remote import denseremotearray


def beta(a, b, size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.beta(a, b)
    else:
        dtype = np.random.beta(a, b, size=1).dtype
        la = denseremotearray.RemoteArray(size, dtype=dtype, dist=dist,
                                        grid_shape=grid_shape, comm=comm)
        la.remote_array[:] = np.random.beta(a, b, size=la.remote_shape)
        return la


def normal(loc=0.0, scale=1.0, size=None, dist=None, grid_shape=None,
           comm=None):
    if size is None:
        return np.random.normal(loc, scale)
    else:
        dtype = np.random.normal(loc, scale, size=1).dtype
        la = denseremotearray.RemoteArray(size, dtype=dtype, dist=dist,
                                        grid_shape=grid_shape, comm=comm)
        la.remote_array[:] = np.random.normal(loc, scale, size=la.remote_shape)
        return la


def rand(size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.rand()
    else:
        dtype = np.random.rand(1).dtype
        la = denseremotearray.RemoteArray(size, dtype=dtype, dist=dist,
                                        grid_shape=grid_shape, comm=comm)
        la.remote_array[:] = np.random.rand(*la.remote_shape)
        return la


def randint(low, high=None, size=None, dist=None, grid_shape=None,
            comm=None):
    if size is None:
        return np.random.randint(low, high)
    else:
        dtype = np.random.randint(low, high, size=1).dtype
        la = denseremotearray.RemoteArray(size, dtype=dtype, dist=dist,
                                        grid_shape=grid_shape, comm=comm)
        la.remote_array[:] = np.random.randint(low, high, size=la.remote_shape)
        return la


def randn(size=None, dist=None, grid_shape=None, comm=None):
    if size is None:
        return np.random.randn()
    else:
        dtype = np.random.randn(1).dtype
        la = denseremotearray.RemoteArray(size, dtype=dtype, dist=dist,
                                        grid_shape=grid_shape, comm=comm)
        la.remote_array[:] = np.random.randn(*la.remote_shape)
        return la
