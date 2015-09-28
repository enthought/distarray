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

    def test_magic_1(self):
        expected = six.b('\x93NUMPY\x01\x00')

        prefix = six.b('\x93NUMPY')
        major = 1
        minor = 0

        result = fmt.magic(major=major, minor=minor, prefix=prefix)
        self.assertEqual(result, expected)


class TestReadMagic(unittest.TestCase):

    def test_read_magic_0(self):
        prefix = six.b('\x93DARRY')
        prefix_len = 8
        fp = six.BytesIO(six.b('\x93DARRY\x03\x02'))

        major, minor = fmt.read_magic(fp, prefix=prefix, prefix_len=prefix_len)

        expected = (3, 2)
        self.assertEqual((major, minor), expected)

    def test_read_magic_1(self):
        prefix = six.b('\x93NUMPY')
        prefix_len = 8
        fp = six.BytesIO(six.b('\x93NUMPY\x01\x01'))

        major, minor = fmt.read_magic(fp, prefix=prefix, prefix_len=prefix_len)

        expected = (1, 1)
        self.assertEqual((major, minor), expected)
