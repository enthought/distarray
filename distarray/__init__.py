# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

from distarray.__version__ import __version__
DISTARRAY_BASE_NAME = '__distarray__'
# intracomm consisting of all the engines, used in MPI mode.
_BASE_COMM = None
# Intercomm left group is client, right group is all engines, used in MPI mode.
INTERCOMM = None
