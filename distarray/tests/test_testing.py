# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from distarray import testing


class TestRaiseTypeError(unittest.TestCase):

    def test_good_dim_dict(self):
        dim_dict = {}
        success, msg = testing.validate_dim_dict(3, dim_dict)
        self.assertTrue(success)

    def test_good_bad_dim_dict(self):
        dim_dict = {'dist_type': 'b'}
        with self.assertRaises(TypeError):
            testing.validate_dim_dict(3, dim_dict)

    def test_good_dim_data(self):
        dim_data = ({}, {}, {})
        success, msg = testing.validate_dim_data(dim_data)
        self.assertTrue(success)

    def test_good_bad_dim_data(self):
        dim_data = ({'dist_type': 'b'}, {}, {})
        with self.assertRaises(TypeError):
            testing.validate_dim_data(dim_data)

    def test_good_distbuffer(self):
        dim_data = ({},)
        distbuffer = dict(__version__='0.10.0',
                          buffer=bytearray([1,2,3,4]),
                          dim_data=dim_data)
        success, msg = testing.validate_distbuffer(distbuffer)
        self.assertTrue(success)

    def test_bad_distbuffer(self):
        dim_data = ({},)
        distbuffer = dict(__venison__='0.10.0',
                          biffer=bytearray([1,2,3,4]),
                          dim_doodle=dim_data)
        with self.assertRaises(TypeError):
            testing.validate_distbuffer(distbuffer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
