import unittest
import numpy as np

from . import validator


class TestValidator(unittest.TestCase):

    def test_block(self):
        dim_data = ({'dist_type': 'b',
            'size': 50,
            'proc_grid_size': 2,
            'proc_grid_rank': 0,
            'start': 0,
            'stop': 10},)
        distbuffer = {'__version__': '1.0.0',
                'buffer': np.ones(10),
                'dim_data': dim_data}

        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_cyclic(self):
        dim_data = ({'dist_type': 'c',
            'size': 50,
            'proc_grid_size': 2,
            'proc_grid_rank': 0,
            'start': 0},)
        distbuffer = {'__version__': '1.0.0',
                'buffer': np.ones(50),
                'dim_data': dim_data}

        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_block_cyclic(self):
        dim_data = ({'dist_type': 'c',
            'size': 50,
            'proc_grid_size': 2,
            'proc_grid_rank': 1,
            'start': 5,
            'block_size': 5},)
        distbuffer = {'__version__': '1.0.0',
                'buffer': np.ones(50),
                'dim_data': dim_data}

        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_unstructured(self):
        dim_data = ({'dist_type': 'u',
            'size': 50,
            'proc_grid_size': 2,
            'proc_grid_rank': 1,
            'indices': np.array([1, 22, 44, 49, 9, 33, 21], dtype=np.uint32)
            },)
        distbuffer = {'__version__': '1.0.0',
                'buffer': np.ones(len(dim_data[0]['indices'])),
                'dim_data': dim_data}

        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_not_distributed(self):
        distbuffer = {'__version__': '1.0.0',
                'buffer': b'blonk',
                'dim_data': ({'dist_type': 'n', 'size': 5},)}
        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_extra_process(self):
        dimdata = {
            'dist_type':'c',
            'size':3,
            'proc_grid_size':4,
            'proc_grid_rank':0,
            'start' : 0,
            }
        distbuffer = {'__version__': '1.0.0',
                'buffer' : b'a',
                'dim_data' : (dimdata,)}
        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)

    def test_empty_process(self):
        dimdata = {
            'dist_type':'c',
            'size':3,
            'proc_grid_size':4,
            'proc_grid_rank':3,
            'start' : 3,
            }
        distbuffer = {'__version__': '1.0.0',
                'buffer' : b'',
                'dim_data' : (dimdata,)}
        is_valid, msg = validator.validate(distbuffer)
        self.assertTrue(is_valid, msg)
