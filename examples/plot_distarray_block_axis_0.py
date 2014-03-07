# encoding: utf-8
# Copyright (c) 2008-2014, IPython Development Team and Enthought, Inc.
"""
Create a simple block distributed distarray, then plot its array
distribution.
"""

__docformat__ = "restructuredtext en"

import distarray
from distarray import plotting


c = distarray.Context()
a = c.zeros((64, 64))
plotting.plot_array_distribution_2d(a)
plotting.show()
