# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plotting functions for distarrays.
"""

from matplotlib import pyplot, colors, cm
from numpy import arange, concatenate, linspace, resize

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


def cmap_discretize(cmap, N):
    """Create a discrete colormap from the continuous colormap cmap.

    Parameters
    ----------
    cmap : colormap instance, or string
        The continuous colormap, as object or name, to make discrete.
        For example, matplotlib.cm.jet, or 'jet'.
    N : int
        The number of discrete colors desired.

    Returns
    -------
    colormap
        The desired discrete colormap.

    Example usage:
    >>> x = resize(arange(100), (5,100))
    >>> djet = cmap_discretize(cm.jet, 5)
    >>> pyplot.imshow(x, cmap=djet)
    """
    # This is copied from:
    # http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations
    if type(cmap) == str:
        cmap = cm.get_cmap(cmap)
    colors_i = concatenate((linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
            for i in xrange(N + 1)
        ]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def plot_array_distribution(darr,
                            title=None,
                            xlabel=None,
                            ylabel=None,
                            yflip=False,
                            legend=False,
                            filename=None,
                            *args, **kwargs):
    """
    Plot a distarray's memory layout. It can be 1D or 2D.
    Elements are colored according to the process they are on.

    Parameters
    ----------
    title : string
        Text label for the plot title, or None.
    xlabel : string
        Text label for the x-axis, or None.
    ylabel : string
        Text label for the y-axis, or None.
    yflip : bool
        If True, then the y-axis increases downwards, to match the layout
        when printing the array itself.
    legend : bool
        If True, then a colorbar legend is drawn to label the colors.
    filename : string
        Output filename for the plot image.

    Returns
    -------
    out
        The memory layout array, as a DistArray.
    """
    # This is based somewhat on:
    #   http://matplotlib.org/examples/api/colorbar_only.html

    out = _get_ranks(darr)
    arr = out.toarray()

    # Coerce to 2D if needed.
    if len(arr.shape) == 1:
        arr.shape = (1, arr.shape[0])

    # Create discrete colormap.
    num_processors = arr.max() + 1
    cmap = cmap_discretize(cm.jet, num_processors)
    bounds = range(num_processors + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the array.
    img = pyplot.matshow(arr, cmap=cmap, norm=norm, *args, **kwargs)

    # Add title and labels.
    if title is not None:
        pyplot.title(title)
    if xlabel is not None:
        pyplot.xlabel(xlabel)
    if ylabel is not None:
        pyplot.ylabel(ylabel)

    # Either invert y-axis, and put tick labels at bottom,
    # or put the x-axis label at the top.
    axis = pyplot.gca()
    if yflip:
        axis.invert_yaxis()
        for tick in axis.xaxis.iter_ticks():
            tick[0].label1On = True
            tick[0].label2On = False
    else:
        axis.xaxis.set_label_position('top')

    # Add colorbar legend.
    if legend:
        cbar = pyplot.colorbar(img)
        cbar_ticks = [0.5 + p for p in range(num_processors)]
        cbar_labels = [str(p) for p in range(num_processors)]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels)
        cbar.set_label('Processor')

    # Adjust size.
    figure = pyplot.gcf()
    figure.set_size_inches(10.0, 5.0)

    # Save to output file.
    if filename is not None:
        pyplot.savefig(filename, dpi=100)

    return out
