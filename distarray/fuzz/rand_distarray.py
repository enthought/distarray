"""
Generate a random distarray with reasonable defaults.
"""

import random

import distarray.dist as da
from distarray.dist import Distribution


def rand_size(max_size=None):
    if max_size is None:
        max_size = 1e3
    return random.randint(1, max_size)


def rand_ndim(max_ndim=None):
    if max_ndim is None:
        max_ndim = 5
    return random.randint(1, max_ndim)


def rand_shape(max_1d=None, ndim=None):
    if ndim is None:
        ndim = rand_ndim()

    if max_1d is None:
        max_1d = 15
    shape = []
    for i in range(ndim):
        shape.append(random.randint(1, max_1d))
    return tuple(shape)


def rand_dist(ndim):
    choices = ('b', 'n', 'c')
    dist = []
    for i in range(ndim):
        dist.append(random.choice(choices))
    return tuple(dist)


def rand_distarray(context):
    shape = rand_shape()
    dist = rand_dist(len(shape))
    db = Distribution.from_shape(context, shape, dist=dist)
    r = da.random.Random(context)
    return r.randint(db, -1e3, 1e3)
