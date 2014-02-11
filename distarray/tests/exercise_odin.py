"""
Test Odin extensions to distarray.

To run these tests, you must have an ipcluster running.
For example,

    $ ipcluster start -n <n> --engines=MPIEngineSetLauncher
"""

import unittest
import numpy as np
from six.moves import zip

from distarray import odin
from distarray.client import Context


@odin.remote
def assert_allclose(da, db):
    assert np.allclose(da, db), "Arrays not equal within tolerance."


@odin.remote
def remote_sin(da):
    return np.sin(da)


@odin.remote
def remote_add50(da):
    return da + 50


@odin.remote
def remote_sum(da):
    return np.sum(da.get_remotearray())


@odin.remote
def remote_add_num(da, num):
    return da + num


@odin.remote
def call_barrier(da):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    return da


@odin.remote
def remote_add_nums(da, num1, num2, num3):
    return da + num1 + num2 + num3


@odin.remote
def remote_add_distarrayproxies(da, dg):
    return da + dg


@odin.remote
def remote_add_mixed(da, num1, dg, num2):
    return da + num1 + dg + num2


@odin.remote
def remote_add_ndarray(da, num, ndarr):
    return da + num + ndarr


@odin.remote
def remote_add_kwargs(da, num1, num2=55):
    return da + num1 + num2


@odin.remote
def remote_add_supermix(da, num1, db, num2, dc, num3=99, num4=66):
    return da + num1 + db + num2 + dc + num3 + num4


@odin.remote
def remote_none(da):
    return None


@odin.remote
def call_remote(da):
    db = remote_add50(da)
    dc = remote_add_num(db, 99)
    return dc


@odin.remote
def parameterless():
    """This is a parameterless function."""
    return None


class TestRemote(unittest.TestCase):

    def setUp(self):
        odin.context._execute('import numpy as np')
        self.da = odin.context.empty((1024, 1024))
        self.da.fill(2 * np.pi)

    def test_remote_sin(self):
        db = remote_sin(self.da)
        assert_allclose(db, 0)

    def test_remote_add50(self):
        dc = remote_add50(self.da)
        assert_allclose(dc, 2 * np.pi + 50)

    def test_remote_sum(self):
        dd = remote_sum(self.da)
        lshapes = self.da.get_remoteshapes()
        expected = [lshape[0] * lshape[1] * (2 * np.pi) for lshape in lshapes]
        for (v, e) in zip(dd, expected):
            self.assertAlmostEqual(v, e, places=5)

    def test_remote_add_num(self):
        de = remote_add_num(self.da, 11)
        assert_allclose(de, 2 * np.pi + 11)

    def test_remote_add_nums(self):
        df = remote_add_nums(self.da, 11, 12, 13)
        assert_allclose(df, 2 * np.pi + 11 + 12 + 13)

    def test_remote_add_distarrayproxies(self):
        dg = odin.context.empty((1024, 1024))
        dg.fill(33)
        dh = remote_add_distarrayproxies(self.da, dg)
        assert_allclose(dh, 33 + 2 * np.pi)

    def test_remote_add_mixed(self):
        di = odin.context.empty((1024, 1024))
        di.fill(33)
        dj = remote_add_mixed(self.da, 11, di, 12)
        assert_allclose(dj, 2 * np.pi + 11 + 33 + 12)

    @unittest.skip('Remotely adding ndarrays not supported.')
    def test_remote_add_ndarray(self):
        shp = self.da.get_remoteshapes()[0]
        ndarr = np.empty(shp)
        ndarr.fill(33)
        dk = remote_add_ndarray(self.da, 11, ndarr)
        assert_allclose(dk, 2 * np.pi + 11 + 33)

    def test_remote_add_kwargs(self):
        dl = remote_add_kwargs(self.da, 11, num2=12)
        assert_allclose(dl, 2 * np.pi + 11 + 12)

    def test_remote_add_supermix(self):
        dm = odin.context.empty((1024, 1024))
        dm.fill(22)
        dn = odin.context.empty((1024, 1024))
        dn.fill(44)
        do = remote_add_supermix(self.da, 11, dm, 33, dc=dn, num3=55)
        expected = 2 * np.pi + 11 + 22 + 33 + 44 + 55 + 66
        assert_allclose(do, expected)

    def test_remote_none(self):
        dp = remote_none(self.da)
        self.assertTrue(dp is None)

    def test_call_remote(self):
        dq = call_remote(self.da)
        assert_allclose(dq, 2 * np.pi + 50 + 99)

    def test_subcontext(self):
        targets = [0, 2]
        subcontext = Context(odin._global_view, targets=targets)
        da = subcontext.empty((1024, 1024))
        da.fill(11)
        with self.assertRaises(ValueError):
            remote_add_num(da, 10)

    def test_barrier(self):
        call_barrier(self.da)

    def test_barrier_with_subcontext(self):
        targets = [0, 2]
        subcontext = Context(odin._global_view, targets=targets)
        da = subcontext.empty((1024, 1024))
        with self.assertRaises(ValueError):
            call_barrier(da)

    def test_parameterless(self):
        rval = parameterless()
        self.assertTrue(rval is None)

    def test_function_metadata(self):
        name = "parameterless"
        docstring = """This is a parameterless function."""
        self.assertEqual(parameterless.__name__, name)
        self.assertEqual(parameterless.__doc__, docstring)


class TestDetermineContext(unittest.TestCase):

    def test_global_context(self):
        da = odin.context.empty((100,))
        db = odin.context.empty((100,))
        self.assertEqual(odin.determine_context(odin.context, (da, db)), odin.context)

    def test_subcontext(self):
        subcontext = Context(odin._global_view, targets=[0, 3])
        da = subcontext.empty((100,))
        db = subcontext.empty((100,))
        self.assertEqual(odin.determine_context(subcontext, (da, db)), subcontext)

    def test_no_proxies(self):
        self.assertEqual(odin.determine_context(odin.context, (11, 12, 'abc')), odin.context)

    def test_mixed_types(self):
        subcontext = Context(odin._global_view, targets=[0, 3])
        da = subcontext.empty((100,))
        self.assertEqual(odin.determine_context(subcontext, (da, 12, 'abc')), da.context)

    def test_mixed_contexts(self):
        subcontext = Context(odin._global_view, targets=[0, 3])
        da = odin.context.empty((100,))
        db = subcontext.empty((100,))
        self.assertRaises(ValueError, odin.determine_context, odin.context, (da, db))


if __name__ == '__main__':
    unittest.main(verbosity=2)
