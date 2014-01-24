from __future__ import print_function

import unittest

import distarray.core.denselocalarray as dla
from distarray.testing import MpiTestCase, comm_null_passes


class TestNDEnumerate(MpiTestCase):

    """Make sure we generate indices compatible with __getitem__."""

    @comm_null_passes
    def test_ndenumerate(self):
        a = dla.LocalArray((16, 16, 2),
                           dist=('c', 'b', None),
                           comm=self.comm)
        for global_inds, value in dla.ndenumerate(a):
            a[global_inds] = 0.0


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
