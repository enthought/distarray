import unittest
import numpy as np

import distarray.core.denselocalarray as dla
from distarray.testing import MpiTestCase, comm_null_passes


class TestFunctions(MpiTestCase):

    @comm_null_passes
    def test_arecompatible(self):
        """Test if two DistArrays are compatible."""
        a = dla.LocalArray((16,16), dtype='int64', comm=self.comm)
        b = dla.LocalArray((16,16), dtype='float32', comm=self.comm)
        self.assertEqual(dla.arecompatible(a,b), True)
        a = dla.LocalArray((16,16), dtype='int64', dist='c', comm=self.comm)
        b = dla.LocalArray((16,16), dtype='float32', dist='b', comm=self.comm)
        self.assertEqual(dla.arecompatible(a,b), False)

    @comm_null_passes
    def test_fromfunction(self):
        """Can we build an array using fromfunction and a trivial function?"""
        def f(*global_inds):
            return 1.0

        a = dla.fromfunction(f, (16, 16), dtype='int64', dist=('b', 'c'),
                             comm=self.comm)
        self.assertEqual(a.shape, (16,16))
        self.assertEqual(a.dtype, np.dtype('int64'))
        for global_inds, value in dla.ndenumerate(a):
            self.assertEqual(1.0, value)

    @comm_null_passes
    def test_fromfunction_complicated(self):
        """Can we build an array using fromfunction and a nontrivial function."""
        def f(*global_inds):
            return sum(global_inds)

        a = dla.fromfunction(f, (16, 16), dtype='int64', dist=('b', 'c'),
                             comm=self.comm)
        self.assertEqual(a.shape, (16,16))
        self.assertEqual(a.dtype, np.dtype('int64'))
        for global_inds, value in dla.ndenumerate(a):
            self.assertEqual(sum(global_inds), value)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
