# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Functions used for tests.
"""

import unittest
import importlib
import tempfile
import os
import types
from uuid import uuid4
from functools import wraps

import numpy as np

from distarray.externals import six
from distarray.externals import protocol_validator
from distarray.globalapi.context import Context, ContextCreationError
from distarray.globalapi.ipython_utils import IPythonClient
from distarray.error import InvalidCommSizeError
from distarray.localapi.mpiutils import MPI, create_comm_of_size


def raise_typeerror(fn):
    """Decorator for protocol validator functions.

    These functions return (success, err_msg), but sometimes we would rather
    have an exception.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        good, msg = fn(*args, **kwargs)
        if not good:
            raise TypeError(msg)
        else:
            return (good, msg)

    return wrapper


validate_dim_dict = raise_typeerror(protocol_validator.validate_dim_dict)
validate_dim_data = raise_typeerror(protocol_validator.validate_dim_data)
validate_distbuffer = raise_typeerror(protocol_validator.validate)


def temp_filepath(extension=''):
    """Return a randomly generated filename.

    This filename is appended to the directory path returned by
    `tempfile.gettempdir()` and has `extension` appended to it.
    """
    tempdir = tempfile.gettempdir()
    filename = str(uuid4())[:8] + extension
    return os.path.join(tempdir, filename)


def import_or_skip(name):
    """Try importing `name`, raise SkipTest on failure.

    Parameters
    ----------
    name : str
        Module name to try to import.

    Returns
    -------
    module : module object
        Module object imported by importlib.

    Raises
    ------
    unittest.SkipTest
        If the attempted import raises an ImportError.

    Examples
    --------
    >>> h5py = import_or_skip('h5py')
    >>> h5py.get_config()
    <h5py.h5.H5PYConfig at 0x103dd5a78>

    """
    try:
        return importlib.import_module(name)
    except ImportError:
        errmsg = '%s not found... skipping.' % name
        raise unittest.SkipTest(errmsg)


def comm_null_passes(fn):
    """Decorator. If `self.comm` is COMM_NULL, pass.

    This allows our tests to pass on processes that have nothing to do.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'comm') and (self.comm == MPI.COMM_NULL):
            pass
        else:
            return fn(self, *args, **kwargs)

    return wrapper


class CommNullPasser(type):

    """Metaclass.

    Applies the `comm_null_passes` decorator to every method on a generated
    class.
    """

    def __new__(cls, name, bases, attrs):

        for attr_name, attr_value in six.iteritems(attrs):
            if isinstance(attr_value, types.FunctionType):
                attrs[attr_name] = comm_null_passes(attr_value)

        return super(CommNullPasser, cls).__new__(cls, name, bases, attrs)


@six.add_metaclass(CommNullPasser)
class ParallelTestCase(unittest.TestCase):

    """Base test class for fully distributed and client-less test cases.

    Overload the `comm_size` class attribute to change the default number of
    processes required.

    Attributes
    ----------
    comm_size : int, default=4
        Indicates how many MPI processes are required for this test to
        run.  If fewer than `comm_size` are available, the test will be
        skipped.
    """

    comm_size = 4

    @classmethod
    def setUpClass(cls):
        try:
            cls.comm = create_comm_of_size(cls.comm_size)
        except InvalidCommSizeError:
            msg = "Must run with comm size >= {}."
            raise unittest.SkipTest(msg.format(cls.comm_size))

    @classmethod
    def tearDownClass(cls):
        if cls.comm != MPI.COMM_NULL:
            cls.comm.Free()


class BaseContextTestCase(unittest.TestCase):

    """Base test class for test cases that use a Context.

    Overload the `ntargets` class attribute to change the default  number of
    engines required.  A `cls.context` object will be created with
    `targets=range(cls.ntargets)`.  Tests will be skipped if there are too
    few targets.

    Attributes
    ----------
    ntargets : int or 'any', default=4
        If an int, indicates how many engines are required for this test to
        run.  If the string 'any', indicates that any number of engines may
        be used with this test.
    """

    ntargets = 4

    @classmethod
    def setUpClass(cls):
        super(BaseContextTestCase, cls).setUpClass()

        # skip if there aren't enough engines

        try:
            if cls.ntargets == 'any':
                cls.context = cls.make_context()
                cls.ntargets = len(cls.context.targets)
            else:
                try:
                    cls.context = cls.make_context(targets=list(range(cls.ntargets)))
                except ValueError:
                    msg = ("Not enough targets available for this test. (%s) "
                        "required" % (cls.ntargets))
                    raise unittest.SkipTest(msg)
        except ContextCreationError as e:
            raise unittest.SkipTest(e.message)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.context.close()
        except RuntimeError:
            pass


class MPIContextTestCase(BaseContextTestCase):

    @classmethod
    def make_context(cls, targets=None):
        return Context(kind='MPI', targets=targets)


class IPythonContextTestCase(BaseContextTestCase):

    @classmethod
    def make_context(cls, targets=None):
        try:
            return Context(kind='IPython', targets=targets)
        except EnvironmentError:
            msg = "You must have an ipcluster running to run this test case."
            raise unittest.SkipTest(msg)

    @classmethod
    def setUpClass(cls):
        super(IPythonContextTestCase, cls).setUpClass()
        cls.client = cls.context.client


class DefaultContextTestCase(BaseContextTestCase):

    @classmethod
    def make_context(cls, targets=None):
        try:
            return Context(targets=targets)
        except EnvironmentError:
            msg = "You must have an ipcluster running to run this test case."
            raise unittest.SkipTest(msg)


def check_targets(required, available):
    """If available < required, raise a SkipTest with a nice error message."""
    if available < required:
        msg = ("This test requires at least {} engines to run; "
               "only {} available.")
        msg = msg.format(required, available)
        raise unittest.SkipTest(msg)


def _assert_localarray_metadata_equal(l0, l1, check_dtype=False):
    np.testing.assert_equal(l0.dist, l1.dist)
    np.testing.assert_equal(l0.global_shape, l1.global_shape)
    np.testing.assert_equal(l0.ndim, l1.ndim)
    np.testing.assert_equal(l0.global_size, l1.global_size)
    np.testing.assert_equal(l0.comm_size, l1.comm_size)
    np.testing.assert_equal(l0.comm_rank, l1.comm_rank)
    np.testing.assert_equal(l0.cart_coords, l1.cart_coords)
    np.testing.assert_equal(l0.grid_shape, l1.grid_shape)
    np.testing.assert_equal(l0.local_shape, l1.local_shape)
    np.testing.assert_equal(l0.local_size, l1.local_size)
    np.testing.assert_equal(l0.ndarray.shape, l1.ndarray.shape)
    if check_dtype:
        np.testing.assert_equal(l0.ndarray.dtype, l1.ndarray.dtype)


def assert_localarrays_allclose(l0, l1, check_dtype=False, rtol=1e-07, atol=0):
    """Call np.testing.assert_allclose on `l0` and `l1`.

    Also, check that LocalArray properties are equal.
    """
    _assert_localarray_metadata_equal(l0, l1, check_dtype=check_dtype)
    np.testing.assert_allclose(l0.ndarray, l1.ndarray, rtol=rtol, atol=atol)


def assert_localarrays_equal(l0, l1, check_dtype=False):
    """Call np.testing.assert_equal on `l0` and `l1`.

    Also, check that LocalArray properties are equal.
    """
    _assert_localarray_metadata_equal(l0, l1, check_dtype=check_dtype)
    np.testing.assert_array_equal(l0.ndarray, l1.ndarray)
