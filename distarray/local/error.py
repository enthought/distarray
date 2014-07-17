# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from distarray.error import DistArrayError


class InvalidBaseCommError(DistArrayError):
    """ Exception class when an object expected to be an MPI.Comm object is not one. """
    pass


class DistError(DistArrayError):
    # Seems to be unused.
    pass


class DistMatrixError(DistArrayError):
    # Seems to be unused.
    pass


class IncompatibleArrayError(DistArrayError):
    """ Exception class when arrays are incompatible. """
    pass


class NullCommError(DistArrayError):
    """ Exception class when an MPI communicator is NULL. """
    pass


class InvalidMapCodeError(DistArrayError):
    # Seems to be unused.
    pass


class InvalidDimensionError(DistArrayError):
    """ Exception class when a specified dimension is invalid. """
    pass
