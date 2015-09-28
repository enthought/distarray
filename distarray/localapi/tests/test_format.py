# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------


import unittest
import six

from distarray.localapi import format as fmt


class TestMagic(unittest.TestCase):

    def test_magic_0(self):
        expected = six.b('\x93DARRY\x03\x02')

        prefix = six.b('\x93DARRY')
        major = 3
        minor = 2

        result = fmt.magic(major=major, minor=minor, prefix=prefix)
        self.assertEqual(result, expected)
