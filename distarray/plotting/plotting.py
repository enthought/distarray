# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plotting functions for distarrays.
"""

__docformat__ = "restructuredtext en"

from matplotlib import pyplot

from distarray.decorators import local


@local
def _get_ranks(arr):
    """
    Given a distarray arr, return a distarray with the same shape, but
    with the elements equal to the rank of the process the element is
    on.
    """
    out = arr.copy()
    out.local_array[:] = arr.comm_rank
    out.local_array = out.local_array.astype(int)
    return out


def plot_array_distribution_2d(darr, *args, **kwargs):
    """
    Plot a 2D distarray's memory layout. Elements are colored according
    to the process they are on.
    """
    out = _get_ranks(darr)
    pyplot.matshow(out.toarray(), *args, **kwargs)
    return out


def show(*args, **kwargs):
    pyplot.show(*args, **kwargs)
