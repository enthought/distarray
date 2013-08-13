import numpy as np
from IPython.parallel import Client
from distarray.client import DistArrayContext
from distarray import odin

import unittest


c = Client()
dv = c[:]
dac = DistArrayContext(dv)


@odin.local(dac)
def local_sin(da):
    return np.sin(da)


@odin.local(dac)
def local_add50(da):
    return da + 50


@odin.local(dac)
def local_sum(da):
    return np.sum(da)


class TestLocal(unittest.TestCase):

    def setUp(self):
        dv.execute('import numpy as np')
        self.da = dac.empty((1024, 1024))
        self.da.fill(2 * np.pi)

    def test_local_sin(self):
        db = local_sin(self.da)

    def test_local_add(self):
        dc = local_add50(self.da)

    def test_local_sum(self):
        dd = local_sum(self.da)


if __name__ == '__main__':
    unittest.main()
