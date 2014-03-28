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
import numpy
import os.path

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


def plot_array_distribution(darray,
                            title=None,
                            xlabel=None,
                            ylabel=None,
                            yflip=False,
                            cell_label=True,
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
    cell_label : bool
        If True, then each cell in the plot is labeled with the array value.
        This can look cluttered for large arrays.
    legend : bool
        If True, then a colorbar legend is drawn to label the colors.
    filename : string
        Output filename for the plot image.

    Returns
    -------
    out
        The process assignment array, as a DistArray.
    """
    # This is based somewhat on:
    #   http://matplotlib.org/examples/api/colorbar_only.html

    # Process per element.
    process_darray = _get_ranks(darray)
    process_array = process_darray.toarray()

    # Values per element.
    values_array = darray.toarray()

    # Coerce to 2D if needed.
    if len(process_array.shape) == 1:
        process_array.shape = (1, process_array.shape[0])
        values_array.shape = process_array.shape

    # Create discrete colormap.
    num_processors = process_array.max() + 1
    cmap = cmap_discretize(cm.jet, num_processors)
    bounds = range(num_processors + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Choose a text color for each processor index.
    # The idea is to pick black for colors near white.
    # This is not sophisticated but ok for this use.
    text_colors = []
    for j in range(num_processors):
        # Get rgb color that matshow() will use.
        jj = float(j + 0.5) / float(num_processors)
        cj = cmap(jj)
        # Get average of rgb values.
        avg = (cj[0] + cj[1] + cj[2]) / 3.0
        # cyan gets avg=0.6111, yellow gets avg=0.6337.
        # Choose cutoff for yellow and not cyan.
        if avg >= 0.625:
            text_color = 'black'
        else:
            text_color = 'white'
        text_colors.append(text_color)

    # Plot the array.
    img = pyplot.matshow(process_array, cmap=cmap, norm=norm, *args, **kwargs)

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

    # Label each cell.
    if cell_label:
        for row in range(values_array.shape[0]):
            for col in range(values_array.shape[1]):
                process = process_array[row, col]
                value = values_array[row, col]
                label = '%d' % (value)
                color = text_colors[process]
                pyplot.text(
                    col, row, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=color)

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

    # Make similar plots for the local arrays...
    plot_local_arrays = True
    local_arrays = darray.get_ndarrays()
    if plot_local_arrays and len(local_arrays) == 4:

        pyplot.clf()
        fig, subfigs = pyplot.subplots(2, 2)
        fig.suptitle('Local Arrays', fontsize=14)
        fig.set_size_inches(10.0, 5.0)

        for ilocal, local_array in enumerate(local_arrays):
            ix, iy = ilocal // 2, ilocal % 2
            sfig = subfigs[ix, iy]

            title = 'Process %d' % (ilocal)
            sfig.set_title(title, fontsize=10)

            # Fill array with the process number so it colors
            # the same as in the global plot.
            plot_local_array = numpy.empty_like(local_array)
            plot_local_array.fill(ilocal)

            shape = local_array.shape

            # I tried to adjust the size of the subplots carefully, with
            # the idea that the size should be proportional to the local array
            # size, but I was not able to work that out.
            # So this makes all the plots the same size which at least
            # does not look too strange.
            img = sfig.imshow(plot_local_array,
                              extent=[-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5],
                              interpolation='nearest',
                              aspect='auto',
                              cmap=cmap, norm=norm,
                              *args, **kwargs)

            # Note that y limits are flipped to get the first row
            # of the arrays at the top of the plot.
            sfig.set_xlim(0 - 0.5, shape[1] - 0.5)
            sfig.set_ylim(shape[0] - 0.5, 0 - 0.5)

            # Configure a grid but otherwise hide the tickmarks.
            x_ticks = [i - 0.5 for i in range(shape[1] + 1)]
            y_ticks = [i - 0.5 for i in range(shape[0] + 1)]
            sfig.xaxis.set_ticks(x_ticks)
            sfig.yaxis.set_ticks(y_ticks)
            sfig.grid(True,
                      linestyle='-',
                      color=text_colors[ilocal])
            all_ticks = []
            all_ticks.extend(sfig.xaxis.iter_ticks())
            all_ticks.extend(sfig.yaxis.iter_ticks())
            for tick in all_ticks:
                tick[0].label1On = False
                tick[0].label2On = False
                tick[0].tick1On = False
                tick[0].tick2On = False

            # Label each cell.
            if cell_label:
                for row in range(shape[0]):
                    for col in range(shape[1]):
                        value = local_array[row, col]
                        label = '%d' % (value)
                        color = text_colors[ilocal]
                        sfig.text(
                            col, row, label,
                            horizontalalignment='center',
                            verticalalignment='center',
                            color=color)

        if filename is not None:
            root, ext = os.path.splitext(filename)
            local_filename = root + '_local' + ext
            pyplot.savefig(local_filename, dpi=100)

    return process_darray
