# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

import unittest
from distarray import metadata_utils


class TestMakeGridShape(unittest.TestCase):

    def test_make_grid_shape(self):
        grid_shape = metadata_utils.make_grid_shape((20, 20), ('b', 'b'), 12)
        self.assertEqual(grid_shape, (3, 4))


class TestPositivify(unittest.TestCase):

    def test_positive_index(self):
        result = metadata_utils.positivify(5, 10)
        self.assertEqual(result, 5)

    def test_negative_index(self):
        result = metadata_utils.positivify(-2, 10)
        self.assertEqual(result, 8)

    def test_out_of_bounds_positive(self):
        with self.assertRaises(IndexError):
            metadata_utils.positivify(11, 10)

    def test_out_of_bounds_negative(self):
        with self.assertRaises(IndexError):
            metadata_utils.positivify(-51, 10)

    def test_positive_slice(self):
        s = slice(5, 7)
        result = metadata_utils.positivify(s, 10)
        self.assertEqual(result, s)

    def test_negative_slice_stop(self):
        s = slice(5, -2)
        result = metadata_utils.positivify(s, 10)
        expected = slice(5, 8)
        self.assertEqual(result, expected)

    def test_no_slice_start(self):
        s = slice(5)
        result = metadata_utils.positivify(s, 10)
        expected = s
        self.assertEqual(result, expected)

    def test_no_slice_stop(self):
        s = slice(5, None)
        result = metadata_utils.positivify(s, 10)
        expected = s
        self.assertEqual(result, expected)

    def test_positive_slice_with_step(self):
        s = slice(5, 7, 2)
        result = metadata_utils.positivify(s, 10)
        expected = s
        self.assertEqual(result, expected)

    def test_negative_slice_with_step(self):
        s = slice(-7, -1, 2)
        result = metadata_utils.positivify(s, 10)
        expected = slice(3, 9, 2)
        self.assertEqual(result, expected)

    def test_out_of_bounds_slice(self):
        s = slice(50, 90)
        result = metadata_utils.positivify(s, 10)
        self.assertEqual(result, s)


class TestSanitizeIndices(unittest.TestCase):

    def test_value_index(self):
        tag, sanitized = metadata_utils.sanitize_indices(10)
        self.assertSequenceEqual(sanitized, (10,))
        self.assertEqual(tag, 'value')

    def test_slice_index(self):
        tag, sanitized = metadata_utils.sanitize_indices(slice(10, 20))
        self.assertSequenceEqual(sanitized, (slice(10, 20),))
        self.assertEqual(tag, 'view')

    def test_tuple_of_values(self):
        tag, sanitized = metadata_utils.sanitize_indices((5, 10))
        self.assertSequenceEqual(sanitized, (5, 10))
        self.assertEqual(tag, 'value')

    def test_tuple_of_slices(self):
        slices = slice(10, 20), slice(20, 30), slice(40, 50)
        tag, sanitized = metadata_utils.sanitize_indices(slices)
        self.assertSequenceEqual(sanitized, slices)
        self.assertEqual(tag, 'view')

    def test_tuple_of_mixed(self):
        slices = slice(10, 20), 25, slice(40, 50)
        tag, sanitized = metadata_utils.sanitize_indices(slices)
        self.assertSequenceEqual(sanitized, slices)
        self.assertEqual(tag, 'view')

    def test_incomplete_indexing_values(self):
        slices = 10, 20, 25, 40, 50
        tag, sanitized = metadata_utils.sanitize_indices(slices, ndim=10)
        self.assertSequenceEqual(sanitized, slices + (slice(None),) * 5)
        self.assertEqual(tag, 'view')

    def test_incomplete_indexing_mixed(self):
        slices = slice(10, 20), 25, slice(40, 50)
        tag, sanitized = metadata_utils.sanitize_indices(slices, ndim=10)
        self.assertSequenceEqual(sanitized, slices + (slice(None),) * 7)
        self.assertEqual(tag, 'view')

    def test_too_many_indices(self):
        with self.assertRaises(IndexError):
            metadata_utils.sanitize_indices((2, 3, 4), ndim=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
