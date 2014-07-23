# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Exception classes for DistArray errors.
"""


class DistArrayError(Exception):
    """ Base exception class for DistArray errors. """
    pass


class MPIDistArrayError(DistArrayError):
    """ Base exception class for MPI distribution errors. """
    pass


class ContextError(DistArrayError):
    """ Exception class when a unique Context cannot be found. """
    pass


class InvalidCommSizeError(MPIDistArrayError):
    """ Exception class when a requested communicator is too large. """
    pass


class InvalidRankError(MPIDistArrayError):
    """ Exception class when an invalid rank is used in a communicator. """
    pass


class DistributionError(DistArrayError):
    """ Exception class when inconsistent distributions are used. """
    pass
