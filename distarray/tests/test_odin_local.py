import numpy as np
from IPython.parallel import Client
from distarray.client import DistArrayContext
from distarray import odin

import unittest


c = Client()
dv = c[:]
dac = DistArrayContext(dv)


@odin.local(dac)
def localsin(da):
    return np.sin(da)


@odin.local(dac)
def localadd50(da):
    return da + 50


@odin.local(dac)
def localsum(da):
    return np.sum(da)


class TestLocal(unittest.TestCase):

    def setUp(self):
        dv.execute('import numpy as np')
        self.da = dac.empty((1024, 1024))
        self.da.fill(2 * np.pi)

    def test_localsin(self):
        db = localsin(self.da)

    def test_localadd(self):
        dc = localadd50(self.da)

    def test_localsum(self):
        dd = localsum(self.da)
        #assert_allclose(db, 0)


if __name__ == '__main__':
    unittest.main()
