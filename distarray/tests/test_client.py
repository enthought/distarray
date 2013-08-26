import unittest
from IPython.parallel import Client
from distarray.client import DistArrayContext


class TestDistArrayContext(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]

    def test_create_DAC(self):
        '''Can we create a plain vanilla context?'''
        dac = DistArrayContext(self.dv)
        self.assertIs(dac.view, self.dv)

    def test_create_DAC_with_targets(self):
        '''Can we create a context with a subset of engines?'''
        dac = DistArrayContext(self.dv, targets=[0, 1])
        self.assertIs(dac.view, self.dv)


if __name__ == '__main__':
    unittest.main(verbosity=2)
