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
# Exports
#----------------------------------------------------------------------------

__all__ = [
    'MPIDistArrayError',
    'InvalidCommSizeError',
    'InvalidRankError',
    'MPICommError']

class MPIDistArrayError(DistArrayError):
    pass

class InvalidCommSizeError(MPIDistArrayError):
    pass

class InvalidRankError(MPIDistArrayError):
    pass

class MPICommError(MPIDistArrayError):
    pass
