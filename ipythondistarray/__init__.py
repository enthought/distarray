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

from ipythondistarray import core
from ipythondistarray.core import *
from ipythondistarray import mpi
from ipythondistarray.mpi import *
from ipythondistarray import random
from ipythondistarray.random import rand, randn

__all__ = []
__all__ += core.__all__
__all__ += mpi.__all__
__all__ += ['rand', 'randn']
