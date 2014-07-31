# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Plotting functions for distarrays.
"""

from matplotlib import pyplot, colors, cm
from numpy import arange, concatenate, empty, linspace, resize

from distarray.externals.six.moves import range


def get_ranks(arr):
    """
    Given a distarray arr, return a distarray with the same shape, but
    with the elements equal to the rank of the process the element is
    on.
    """
    from distarray.localapi import LocalArray
    out = LocalArray(distribution=arr.distribution, dtype=int)
    out.fill(arr.comm_rank)
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

    Example
    -------
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
            for i in range(N + 1)
        ]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def create_discrete_colormaps(num_values):
    """ Create colormap objects for a discrete colormap.

    Parameters
    ----------
    num_values : The number of distinct colors to use.

    Returns
    -------
    cmap, norm, text_colors : tuple
        The matplotlib colormap, norm, and recommended text colors.
        text_colors is an array of length num_values,
        with each entry being a nice color for text drawn
        on top of the colormap selection.
    """
    # Create discrete colormap for matplotlib.
    cmap = cmap_discretize(cm.jet, num_values)
    bounds = range(num_values + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Choose a text color for each discrete color.
    # The idea is to pick black for colors near white.
    # This is not sophisticated but ok for this use.
    text_colors = []
    for j in range(num_values):
        # Get rgb color that matshow() will use.
        jj = float(j + 0.5) / float(num_values)
        cj = cmap(jj)
        # Get average of rgb values.
        avg = (cj[0] + cj[1] + cj[2]) / 3.0
        # With 4-color jet, avg cyan=0.6111, yellow=0.6337.
        # Choose empirically reasonable cutoff.
        if avg >= 0.5:
            text_color = 'black'
        else:
            text_color = 'white'
        text_colors.append(text_color)

    # Return a tuple with all the parts.
    colormaps = (cmap, norm, text_colors)
    return colormaps


def plot_local_array_subfigure(subfig,
                               local_array,
                               process,
                               coord,
                               colormap_objects,
                               *args, **kwargs):
    """ Plot a single local_array into a matplotlib subfigure. """
    title = 'Process %r' % (coord,)
    subfig.set_title(title, fontsize=10)

    # Coerce to 2D if needed.
    if len(local_array.shape) == 1:
        local_array.shape = (1, local_array.shape[0])

    # Fill array with the process number.
    # (Then it will color the same as in the global plot.)
    shape = local_array.shape
    plot_array = empty(shape, dtype=int)
    plot_array.fill(process)

    cmap, norm, text_colors = colormap_objects
    text_color = text_colors[process]

    # I tried to adjust the size of the subplots carefully, with
    # the idea that the size should be proportional to the local array
    # size, but I was not able to work that out.
    # So this makes all the plots the same size which at least
    # does not look too strange.
    extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
    subfig.imshow(plot_array,
                  extent=extent,
                  interpolation='nearest',
                  aspect='auto',
                  cmap=cmap, norm=norm,
                  *args, **kwargs)

    # Note that y limits are flipped to get the first row
    # of the arrays at the top of the plot.
    subfig.set_xlim(0 - 0.5, shape[1] - 0.5)
    subfig.set_ylim(shape[0] - 0.5, 0 - 0.5)

    # Configure a grid but otherwise hide the tickmarks.
    x_ticks = [i - 0.5 for i in range(shape[1] + 1)]
    y_ticks = [i - 0.5 for i in range(shape[0] + 1)]
    subfig.xaxis.set_ticks(x_ticks)
    subfig.yaxis.set_ticks(y_ticks)
    subfig.grid(True, linestyle='-', color=text_color)
    all_ticks = []
    all_ticks.extend(subfig.xaxis.iter_ticks())
    all_ticks.extend(subfig.yaxis.iter_ticks())
    for tick in all_ticks:
        tick[0].label1On = False
        tick[0].label2On = False
        tick[0].tick1On = False
        tick[0].tick2On = False

    # Label each cell.
    for row in range(shape[0]):
        for col in range(shape[1]):
            value = local_array[row, col]
            label = '%d' % (value)
            subfig.text(
                col, row, label,
                horizontalalignment='center',
                verticalalignment='center',
                color=text_color)


def plot_local_arrays(darray,
                      process_coords,
                      colormap_objects,
                      filename):
    """ Plot the local arrays as a multi-figure matplotlib plot. """
    # Get the local arrays that are not empty.
    ndarrays = darray.get_ndarrays()
    local_arrays = []
    for processor, local_array in enumerate(ndarrays):
        processor_coord = process_coords[processor]
        if local_array.size > 0:
            local_arrays.append((processor, processor_coord, local_array))

    pyplot.clf()

    num_local_arrays = len(local_arrays)
    if (num_local_arrays % 2) == 0:
        # 2 X N grid
        subplot_grid = (2, num_local_arrays // 2)
    else:
        # N x 1 grid
        subplot_grid = (num_local_arrays, 1)

    _, subfigs = pyplot.subplots(*subplot_grid)
    for i, (process, coord, local_array) in enumerate(local_arrays):
        if subplot_grid[1] > 1:
            N = subplot_grid[1]
            row, col = i // N, i % N
            subfig = subfigs[row, col]
        else:
            subfig = subfigs[i]
        plot_local_array_subfigure(subfig,
                                   local_array,
                                   process,
                                   coord,
                                   colormap_objects)

    # Add main title and adjust size.
    figure = pyplot.gcf()
    figure.suptitle('Local Arrays', fontsize=14)
    figure.set_size_inches(10.0, 5.0)

    if filename is not None:
        pyplot.savefig(filename, dpi=100)


def plot_array_distribution(darray,
                            process_coords,
                            title=None,
                            xlabel=None,
                            ylabel=None,
                            yflip=False,
                            cell_label=True,
                            legend=False,
                            global_plot_filename=None,
                            local_plot_filename=None,
                            *args, **kwargs):
    """
    Plot a distarray's memory layout. It can be 1D or 2D.
    Elements are colored according to the process they are on.

    Parameters
    ----------
    darray : DistArray
        The distributed array to plot.
    process_coords : List of tuples.
        The process grid coordinates.
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
    global_plot_filename : string
        Output filename for the global array plot image.
    local_plot_filename : string
        Output filename for the local array plot image.

    Returns
    -------
    out
        The process assignment array, as a DistArray.
    """
    # This is based somewhat on:
    #   http://matplotlib.org/examples/api/colorbar_only.html

    # Process per element.
    ctx = darray.context
    ctx.register(get_ranks)
    process_darray = ctx.get_ranks(darray)
    process_array = process_darray.toarray()

    # Values per element.
    values_array = darray.toarray()

    # Coerce to 2D if needed.
    if len(process_array.shape) == 1:
        process_array.shape = (1, process_array.shape[0])
        values_array.shape = process_array.shape

    # Create discrete colormap.
    num_processors = int(process_array.max()) + 1
    colormap_objects = create_discrete_colormaps(num_processors)
    cmap, norm, text_colors = colormap_objects

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
    if global_plot_filename is not None:
        pyplot.savefig(global_plot_filename, dpi=100)

    # Make similar plots for the local arrays...
    if local_plot_filename is not None:
        plot_local_arrays(darray,
                          process_coords,
                          colormap_objects,
                          local_plot_filename)

    return process_darray
