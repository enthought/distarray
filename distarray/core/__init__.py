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

from distarray.core import densedistarray
from distarray.core.densedistarray import *

from distarray.core import base
from distarray.core.base import *

from distarray.core import benchmark
from distarray.core.benchmark import *

__all__ = []
__all__ += densedistarray.__all__
__all__ += base.__all__
__all__ += benchmark.__all__
