# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
Define error classes.
"""


class DistArrayError(Exception):
    pass


class MPIDistArrayError(DistArrayError):
    pass


class ContextError(DistArrayError):
    pass


class InvalidCommSizeError(MPIDistArrayError):
    pass


class InvalidRankError(MPIDistArrayError):
    pass


class MPICommError(MPIDistArrayError):
    pass


class DistributionError(DistArrayError):
    pass
