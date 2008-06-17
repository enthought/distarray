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

from ipythondistarray.core import distarray
from ipythondistarray.core.distarray import *

from ipythondistarray.core import base
from ipythondistarray.core.base import *

__all__ = []
__all__ += distarray.__all__
__all__ += base.__all__
