# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Modules dealing with the global index-space view of `DistArray`\s.

In other words, the view from the client.
"""

from __future__ import absolute_import

from distarray.globalapi.distarray import DistArray
from distarray.globalapi.context import Context, ContextCreationError
from distarray.globalapi.maps import Distribution
from distarray.globalapi.functions import *
