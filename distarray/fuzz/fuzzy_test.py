import random

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal)

import distarray.dist as dist
from distarray.dist.decorators import local, vectorize


def fuzzy_test(darr):
    ndarr = darr.toarray()

    def assert_dist_numpy_equal(darr, ndarr):
        if darr is None and ndarr is None:
            return
        elif isinstance(darr, dist.DistArray):
            assert_array_almost_equal(darr.toarray(), ndarr)
        assert_almost_equal(darr, ndarr)

    def test_method(darr, ndarr, meth_name, args=None, kwargs=None):
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        dist_meth = getattr(darr, meth_name)
        nd_meth = getattr(ndarr, meth_name)
        assert_dist_numpy_equal(dist_meth(*args, **kwargs),
                                nd_meth(*args, **kwargs))

    def random_index(darr):
        shape = darr.shape
        index = []
        for i in range(len(shape)):
            index.append(random.choice(range(shape[i])))
        return tuple(index)

    # test some reductions
    test_method(darr, ndarr, 'sum')
    test_method(darr, ndarr, 'var')
    test_method(darr, ndarr, 'mean')
    # test getting and setting
    test_method(darr, ndarr, '__getitem__', args=(random_index(darr),))
    test_method(darr, ndarr, '__setitem__', args=(random_index(darr),
                                                  random.random()))
    # test a binary function
    assert_array_almost_equal(dist.add(darr, darr).toarray(), np.add(ndarr,
                                                                     ndarr))
    # test a unary function
    assert_array_almost_equal(dist.cos(darr).toarray(), np.cos(ndarr))

    # test local
    @local
    def local_add50(darr):
        return darr + 50
    assert_array_almost_equal(local_add50(darr).toarray(), ndarr + 50)

    # test vectorize
    @vectorize
    def sqr_add10(a):
        return a**2 + 10
    assert_array_almost_equal(sqr_add10(darr).toarray(), ndarr**2 + 10)
