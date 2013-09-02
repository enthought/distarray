import unittest
from IPython.parallel import Client
from distarray.client import DistArrayContext

class TestClient(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        if len(self.dv.targets) < 4:
            raise unittest.SkipTest('Must set up a cluster with at least 4 engines running.')

    def testCreateDAC(self):
        '''Can we create a plain vanilla context?'''
        dac = DistArrayContext(self.dv)
        self.assertIs(dac.view, self.dv)

    def testCreateDACwithTargets(self):
        '''Can we create a context with a subset of engines?'''
        dac = DistArrayContext(self.dv, targets=[0,1])
        self.assertIs(dac.view, self.dv)

    def testCreateDACwithSubView(self):
        '''Context's view must encompass all ranks in the MPI communicator.'''
        subview = self.client[:1]
        with self.assertRaises(ValueError):
            dac = DistArrayContext(subview)

    def testCreateDACwithTargetsRanks(self):
        '''Is the ranks attribute of a Context object contiguous?'''
        targets = [2,3]
        dac = DistArrayContext(self.dv, targets=targets)
        self.assertEqual(set(dac.ranks), set(range(len(targets))))
