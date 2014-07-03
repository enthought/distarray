# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Create a simple block distributed distarray, then plot its array
distribution.
"""

from matplotlib import pyplot

from distarray import plotting
from distarray.dist import Context, Distribution


c = Context()
d = Distribution(c, (64, 64))
a = c.zeros(d)
process_coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
plotting.plot_array_distribution(a, process_coords, cell_label=False,
                                 legend=True)
pyplot.show()
