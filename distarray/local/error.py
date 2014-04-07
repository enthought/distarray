# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from distarray.error import DistArrayError


class InvalidBaseCommError(DistArrayError):
    pass


class DistError(DistArrayError):
    pass


class DistMatrixError(DistArrayError):
    pass


class IncompatibleArrayError(DistArrayError):
    pass


class NullCommError(DistArrayError):
    pass


class InvalidMapCodeError(DistArrayError):
    pass


class InvalidDimensionError(DistArrayError):
    pass
