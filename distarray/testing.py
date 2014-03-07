import unittest
import importlib
import tempfile
import os
from uuid import uuid4
from functools import wraps

from distarray.error import InvalidCommSizeError
from distarray.mpiutils import MPI, create_comm_of_size


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
    """Decorator. If `self.comm` is COMM_NULL, pass."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.comm == MPI.COMM_NULL:
            pass
        else:
            return fn(self, *args, **kwargs)

    return wrapper


class MpiTestCase(unittest.TestCase):

    """Base test class for MPI test cases.

    Overload `get_comm_size` to change the default comm size (default is 4).
    Overload `more_setUp` to add more to the default `setUp`.
    Overload `more_tearDown` to add more to the default `tearDown`.
    """

    def get_comm_size(self):
        return 4

    def more_setUp(self):
        pass

    def setUp(self):
        try:
            self.comm = create_comm_of_size(self.get_comm_size())
        except InvalidCommSizeError:
            msg = "Must run with comm size >= {}."
            raise unittest.SkipTest(msg.format(self.get_comm_size()))
        else:
            self.more_setUp()

    def more_tearDown(self):
        pass

    def tearDown(self):
        self.more_tearDown()
        if self.comm != MPI.COMM_NULL:
            self.comm.Free()
