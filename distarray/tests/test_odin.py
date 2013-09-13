"""
Test Odin extensions to distarray.

To run these tests, you must have an ipcluster running
For example,

    $ ipcluster start -n <n> --engines=MPIEngineSetLauncher
"""

import unittest
import numpy as np

from distarray import odin
from distarray.client import DistArrayContext


@odin.local
def assert_allclose(da, db):
    assert np.allclose(da, db), "Arrays not equal within tolerance."


@odin.local
def local_sin(da):
    return np.sin(da)


@odin.local
def local_add50(da):
    return da + 50


@odin.local
def local_sum(da):
    return np.sum(da.get_localarray())


@odin.local
def local_add_num(da, num):
    return da + num


@odin.local
def call_barrier(da):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    return da


@odin.local
def local_add_nums(da, num1, num2, num3):
    return da + num1 + num2 + num3


@odin.local
def local_add_distarrayproxies(da, dg):
    return da + dg


@odin.local
def local_add_mixed(da, num1, dg, num2):
    return da + num1 + dg + num2


@odin.local
def local_add_ndarray(da, num, ndarr):
    return da + num + ndarr


@odin.local
def local_add_kwargs(da, num1, num2=55):
    return da + num1 + num2


@odin.local
def local_add_supermix(da, num1, db, num2, dc, num3=99, num4=66):
    return da + num1 + db + num2 + dc + num3 + num4


@odin.local
def local_none(da):
    return None


@odin.local
def call_local(da):
    db = local_add50(da)
    dc = local_add_num(db, 99)
    return dc


class TestLocal(unittest.TestCase):

    def setUp(self):
        odin.context._execute('import numpy as np')
        self.da = odin.context.empty((1024, 1024))
        self.da.fill(2 * np.pi)

    def test_local_sin(self):
        db = local_sin(self.da)
        assert_allclose(db, 0)

    def test_local_add(self):
        dc = local_add50(self.da)
        assert_allclose(dc, 2 * np.pi + 50)

    def test_local_sum(self):
        dd = local_sum(self.da)
        client_dd = np.array(odin.context._pull(dd.key))

        shape = self.da.get_localshapes()[0]
        sum_val = shape[0] * shape[1] * (2 * np.pi)

        assert_allclose(client_dd, sum_val)

    def test_local_add_num(self):
        de = local_add_num(self.da, 11)
        assert_allclose(de, 2 * np.pi + 11)

    def test_local_add_nums(self):
        df = local_add_nums(self.da, 11, 12, 13)
        assert_allclose(df, 2 * np.pi + 11 + 12 + 13)

    def test_local_add_distarrayproxies(self):
        dg = odin.context.empty((1024, 1024))
        dg.fill(33)
        dh = local_add_distarrayproxies(self.da, dg)
        assert_allclose(dh, 33 + 2 * np.pi)

    def test_local_add_mixed(self):
        di = odin.context.empty((1024, 1024))
        di.fill(33)
        dj = local_add_mixed(self.da, 11, di, 12)
        assert_allclose(dj, 2 * np.pi + 11 + 33 + 12)

    @unittest.skip('Locally adding ndarrays not supported.')
    def test_local_add_ndarray(self):
        shp = self.da.get_localshapes()[0]
        ndarr = np.empty(shp)
        ndarr.fill(33)
        dk = local_add_ndarray(self.da, 11, ndarr)
        assert_allclose(dk, 2 * np.pi + 11 + 33)

    def test_local_add_kwargs(self):
        dl = local_add_kwargs(self.da, 11, num2=12)
        assert_allclose(dl, 2 * np.pi + 11 + 12)

    def test_local_add_supermix(self):
        dm = odin.context.empty((1024, 1024))
        dm.fill(22)
        dn = odin.context.empty((1024, 1024))
        dn.fill(44)
        do = local_add_supermix(self.da, 11, dm, 33, dc=dn, num3=55)
        expected = 2 * np.pi + 11 + 22 + 33 + 44 + 55 + 66
        assert_allclose(do, expected)

    @unittest.skip("Doesn't know what to do with NoneType")
    def test_local_none(self):
        dp = local_none(self.da)
        client_dp = odin.context._pull(dp.key)
        print client_dp

    def test_call_local(self):
        dq = call_local(self.da)
        assert_allclose(dq, 2 * np.pi + 50 + 99)

    def test_subcontext(self):
        targets = [0, 2]
        subcontext = DistArrayContext(odin._global_view, targets=targets)
        da = subcontext.empty((1024, 1024))
        da.fill(11)
        de = local_add_num(da, 10)
        assert_allclose(de, 11 + 10)

    def test_barrier(self):
        call_barrier(self.da)

    def test_barrier_with_subcontext(self):
        targets = [0, 2]
        subcontext = DistArrayContext(odin._global_view, targets=targets)
        da = subcontext.empty((1024, 1024))
        da.fill(11)
        call_barrier(da)

class TestUtils(unittest.TestCase):

    def test_flatten(self):
        self.assertEqual(odin.flatten(zip([1, 2, 3], [4, 5, 6])),
                                      [1, 4, 2, 5, 3, 6])

    def test_all_equal_false(self):
        self.assertFalse(odin.all_equal([1, 2, 3, 4, 5]))

    def test_all_equal_true(self):
        self.assertTrue(odin.all_equal([7, 7, 7, 7, 7]))

    def test_all_equal_contexts(self):
        subcontext = DistArrayContext(odin._global_view, targets=[0, 3])
        db = subcontext.empty((100,))
        dc = subcontext.ones((100,))
        contexts = (subcontext, db.context, dc.context)
        self.assertTrue(odin.all_equal(contexts))


class TestDetermineContext(unittest.TestCase):

    def test_global_context(self):
        da = odin.context.empty((100,))
        db = odin.context.empty((100,))
        self.assertEqual(odin.determine_context((da, db)), odin.context)

    def test_subcontext(self):
        subcontext = DistArrayContext(odin._global_view, targets=[0, 3])
        da = subcontext.empty((100,))
        db = subcontext.empty((100,))
        self.assertEqual(odin.determine_context((da, db)), subcontext)

    def test_no_proxies(self):
        self.assertEqual(odin.determine_context((11, 12, 'abc')), odin.context)

    def test_mixed_types(self):
        subcontext = DistArrayContext(odin._global_view, targets=[0, 3])
        da = subcontext.empty((100,))
        self.assertEqual(odin.determine_context((da, 12, 'abc')), da.context)

    def test_mixed_contexts(self):
        subcontext = DistArrayContext(odin._global_view, targets=[0, 3])
        da = odin.context.empty((100,))
        db = subcontext.empty((100,))
        self.assertRaises(ValueError, odin.determine_context, (da, db))

if __name__ == '__main__':
    unittest.main(verbosity=2)
