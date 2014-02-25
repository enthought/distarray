import unittest
from functools import wraps

from distarray.error import InvalidCommSizeError
from distarray.mpiutils import MPI, create_comm_of_size


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

    Overload `get_comm_size` to change the default comm size (default is
    4).  Overload `more_setUp` to add more to the default `setUp`.
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

    def tearDown(self):
        if self.comm != MPI.COMM_NULL:
            self.comm.Free()
