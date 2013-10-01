import unittest
from IPython.parallel import Client
from distarray.client import Context

class TestClient(unittest.TestCase):

    def setUp(self):
        self.client = Client()
        self.dv = self.client[:]
        if len(self.dv.targets) < 4:
            raise unittest.SkipTest('Must set up a cluster with at least 4 engines running.')

    def testCreateDAC(self):
        '''Can we create a plain vanilla context?'''
        dac = Context(self.dv)
        self.assertIs(dac.view, self.dv)

    def testCreateDACwithTargets(self):
        '''Can we create a context with a subset of engines?'''
        dac = Context(self.dv, targets=[0,1])
        self.assertIs(dac.view, self.dv)

    def testCreateDACwithSubView(self):
        '''Context's view must encompass all ranks in the MPI communicator.'''
        subview = self.client[:1]
        with self.assertRaises(ValueError):
            dac = Context(subview)

    def testCreateDACwithTargetsRanks(self):
        '''Check that the target <=> rank mapping is consistent.'''
        targets = [3,2]
        dac = Context(self.dv, targets=targets)
        self.assertEqual(set(dac.targets), set(dac.target_to_rank.keys()))
        self.assertEqual(set(range(len(dac.targets))), set(dac.target_to_rank.values()))
