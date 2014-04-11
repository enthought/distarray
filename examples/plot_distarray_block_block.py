# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Create a distarray, then plot its array distribution.
"""

from matplotlib import pyplot

import distarray
from distarray import plotting


c = distarray.Context()
a = c.zeros((64, 64), dtype='int32', dist=('b', 'b'))
process_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
plotting.plot_array_distribution(a, process_coords, cell_label=False,
                                 legend=True)
pyplot.show()
