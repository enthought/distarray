# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------


class DistArrayError(Exception):
    pass


class MPIDistArrayError(DistArrayError):
    pass


class InvalidCommSizeError(MPIDistArrayError):
    pass


class InvalidRankError(MPIDistArrayError):
    pass


class MPICommError(MPIDistArrayError):
    pass
