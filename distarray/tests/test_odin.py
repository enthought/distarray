import numpy as np
from IPython.parallel import Client
from distarray.client import DistArrayContext
from distarray import odin

import unittest


# To run these tests, you must have an ipcluster running
# For example, `ipcluster -n 4


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


@odin.local(dac)
def local_add_num(da, num):
    return da + num


@odin.local(dac)
def local_add_nums(da, num1, num2, num3):
    return da + num1 + num2 + num3


@odin.local(dac)
def local_add_distarrayproxies(da, dg):
    return da + dg


@odin.local(dac)
def local_add_mixed(da, num1, dg, num2):
    return da + num1 + dg + num2


@odin.local(dac)
def local_add_ndarray(da, num, ndarr):
    return da + num + ndarr


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

    def test_local_add_num(self):
        de = local_add_num(self.da, 11)

    def test_local_add_nums(self):
        df = local_add_nums(self.da, 11, 12, 13)

    def test_local_add_distarrayproxies(self):
        dg = dac.empty((1024, 1024))
        dg.fill(33)
        dh = local_add_distarrayproxies(self.da, dg)

    def test_local_add_mixed(self):
        di = dac.empty((1024, 1024))
        di.fill(33)
        dj = local_add_mixed(self.da, 11, di, 12)

    def test_local_add_ndarray(self):
        shp = self.da.get_localshapes()[0]
        ndarr = np.empty(shp)
        ndarr.fill(33)
        dj = local_add_ndarray(self.da, 11, ndarr)


if __name__ == '__main__':
    unittest.main()
