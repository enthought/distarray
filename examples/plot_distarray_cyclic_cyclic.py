# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Create a distarray, then plot its array distribution.
"""

import distarray
from distarray import plotting


c = distarray.Context()
a = c.zeros((64, 64), dtype='int32', dist=('c', 'c'))
plotting.plot_array_distribution_2d(a)
plotting.show()
