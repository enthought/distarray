# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

from distarray.error import DistArrayError


#----------------------------------------------------------------------------
# Exceptions
#----------------------------------------------------------------------------

class InvalidBaseCommError(DistArrayError):
    pass

class InvalidGridShapeError(DistArrayError):
    pass

class GridShapeError(DistArrayError):
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
