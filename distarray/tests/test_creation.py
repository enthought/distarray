import numpy
from numpy.testing import assert_array_equal, assert_allclose

from distarray.creation import ones, empty, zeros, fromndarray, from_dim_data
from distarray.context import Context
from distarray.testing import IpclusterTestCase


class TestDistArrayCreation(IpclusterTestCase):
    """Test distarray creation methods"""

    def setUp(self):
        self.context = Context(self.client)

     # overloads base class...
    def tearDown(self):
        del self.context
        super(TestDistArrayCreation, self).tearDown()

    def test_zeros(self):
        shape = (16, 16)
        zero_distarray = zeros(shape, context=self.context)
        zero_ndarray = numpy.zeros(shape)
        assert_array_equal(zero_distarray.tondarray(), zero_ndarray)

    def test_ones(self):
        shape = (16, 16)
        one_distarray = ones(shape, context=self.context)
        one_ndarray = numpy.ones(shape)
        assert_array_equal(one_distarray.tondarray(), one_ndarray)

    def test_empty(self):
        shape = (16, 16)
        empty_distarray = empty(shape, context=self.context)
        self.assertEqual(empty_distarray.shape, shape)

    def test_fromndarray(self):
        ndarr = numpy.arange(16).reshape(4, 4)
        distarr = fromndarray(ndarr, context=self.context)
        for (i, j), val in numpy.ndenumerate(ndarr):
            self.assertEqual(distarr[i, j], ndarr[i, j])

    def test_from_dim_data_1d(self):
        total_size = 40
        ddpp = [
            ({'dist_type': 'u',
              'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'size': total_size},),
            ({'dist_type': 'u',
              'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'size': total_size},)]
        distarr = from_dim_data(ddpp, context=self.context)
        for i in range(total_size):
            distarr[i] = i
        localarrays = distarr.get_localarrays()
        for i, arr in enumerate(localarrays):
            assert_allclose(arr, ddpp[i][0]['indices'])

    def test_from_dim_data_bu(self):
        rows = 9
        cols = 10
        col_indices = numpy.random.permutation(range(cols))
        ddpp = [
             (
              {'dist_type': 'b',
               'start': 0,
               'stop': rows // 2,
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:len(col_indices)//3],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': 0,
               'stop': rows // 2,
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[len(col_indices)//3:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': rows//2,
               'stop': rows,
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:len(col_indices)//3],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'b',
               'start': rows//2,
               'stop': rows,
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[len(col_indices)//3:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             )]
        distarr = from_dim_data(ddpp, context=self.context)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j

    def test_from_dim_data_uu(self):
        rows = 6
        cols = 20
        row_indices = numpy.random.permutation(range(rows))
        col_indices = numpy.random.permutation(range(cols))
        ddpp = [
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows//2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols//4],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows//2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols//4:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows//2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols//4],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows//2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols//4:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             )]
        distarr = from_dim_data(ddpp, context=self.context)
        for i in range(rows):
            for j in range(cols):
                distarr[i, j] = i*cols + j
