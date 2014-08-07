# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Plot DistArray distributions for the Distributed Array Protocol documentation.

The output from this script should be directly usable in the Distributed Array
Protocol documentation (if saved as examples.rst).  The output .png files
should be copied to the images folder as well.

To generate the documentation, run:

    $ python plot_distarray_protocol.py > examples.rst

This should create both the ``examples.rst`` RestructuredText file and many
.png files in the ``images`` subdirectory.

All of these can be copied to the corresponding location in the
``distributed-array-protocol`` directory tree, and the sphinx documentation can
be rebuilt with ``make html``.
"""

from __future__ import print_function

import os.path
from pprint import pformat
from numpy import empty
from numpy.random import permutation, seed

import distarray
from distarray import plotting
from distarray.globalapi import Distribution


def print_array_documentation(context, array, title, text,
                              global_plot_filename, local_plot_filename):
    """Print some properties of the array.

    The output is rst formatted so that it can be directly used for
    documentation. It includes a plot of the array distribution, some
    properties that are same for each process, and properties that vary per
    process.

    Parameters
    ----------
    context : distarray.globalapi.Context
        The context that will be used to query the array properties.
    array : DistArray
        The array to describe.
    title : string
        The document section title.
    text : string
        A description of the array layout to add to the document.
    global_plot_filename : string
        The filename of the global array figure to add to the document.
    local_plot_filename : string
        The filename of the local array figure to add to the document.
    """

    def rst_lines(obj):
        """ Return lines of text that format obj for an .rst document. """
        text = pformat(obj)
        lines = text.split('\n')
        # pformat() gives blank lines for 3d arrays, which confuse Sphinx.
        trims = [line for line in lines if len(line) > 0]
        return trims

    def rst_code(lines):
        """Format a list of lines into a python code block.

        Returns a list of text lines.
        """
        code_lines = []
        code_lines.append("")
        code_lines.append(".. code-block:: python")
        code_lines.append("")
        for line in lines:
            code_line = "    " + line
            code_lines.append(code_line)
        code_lines.append("")
        return code_lines

    def rst_print(obj):
        """ Return text that formats obj nicely for an .rst document. """
        lines = rst_lines(obj)
        text = '\n'.join(lines)
        return text

    def rst_plot(filename):
        """ Reference a plot in the .rst document.

        The plot must be created elsewhere, this does not make it.
        The path emitted assumes some organization of the
        documentation directory.
        """
        print(".. image:: %s" % (filename))
        print()

    def text_block_size(lines):
        """ Determine the number of rows and columns to print lines.

        Parameters
        ----------
        lines : list of text strings

        Returns
        -------
        line_count, line_width : integers
            The number of lines and columns required to contain the text.
        """
        line_count = len(lines)
        line_width = max([len(line) for line in lines])
        return line_count, line_width

    def text_block_max_size(lines_list):
        """ Determine  number of rows/columns for the largest line list.

        Parameters
        ----------
        lines_list : list of list of text strings
            Each entry in the outer list is termed a 'block'.
            Each block, which is a list of text strings,
            needs some size of space R x C to fit.

        Returns
        -------
            The text box size, in lines and columns, which
            is just large enough to display all of the blocks.
        """
        # Get line count and width needed for each block.
        block_size = empty((len(lines_list), 2), dtype=int)
        for itext, lines in enumerate(lines_list):
            line_count, line_width = text_block_size(lines)
            block_size[itext, 0] = line_count
            block_size[itext, 1] = line_width
        # Get maximum which is enough to hold any of them.
        max_size = block_size.max(axis=0)
        max_rows, max_cols = max_size[0], max_size[1]
        return max_rows, max_cols

    def rst_print_lines(lines_list):
        """ Print the list of lines. """
        for lines in lines_list:
            for line in lines:
                print(line)

    def rst_table(rows, cols, lines_list):
        """ Print the list of lines as a .rst table. """
        num_cells = rows * cols
        num_texts = len(lines_list)
        if num_cells != num_texts:
            raise ValueError('Invalid table size %d x %d for %d entries.' % (
                rows, cols, num_texts))
        # Determine table size needed for biggest text blocks.
        max_lines, max_cols = text_block_max_size(lines_list)
        # Sphinx table row separator.
        sep = '-' * max_cols
        seps = [sep for i in range(cols)]
        header = '+' + '+'.join(seps) + '+'
        # Group text blocks into array pattern.
        print(header)
        for row in range(rows):
            for line in range(max_lines):
                col_lines = []
                for col in range(cols):
                    iblock = row * cols + col
                    lines = lines_list[iblock]
                    if line < len(lines):
                        col_line = lines[line]
                    else:
                        col_line = ''
                    col_line = col_line.ljust(max_cols)
                    col_lines.append(col_line)
                text = '|' + '|'.join(col_lines) + '|'
                print(text)
            print(header)
        print()

    # Examine the array on all the engines.
    def _array_attrs(local_arr):
        distbuffer = local_arr.__distarray__()
        return dict(db_keys=list(distbuffer.keys()),
                    db_version=distbuffer["__version__"],
                    db_buffer=distbuffer["buffer"],
                    db_dim_data=distbuffer["dim_data"],
                    db_coords=local_arr.cart_coords,
                    )

    attrs = context.apply(_array_attrs, (array.key,),
                          targets=array.targets)

    db_keys = [a['db_keys'] for a in attrs]
    db_version = [a['db_version'] for a in attrs]
    db_buffer = [a['db_buffer'] for a in attrs]
    db_dim_data = [a['db_dim_data'] for a in attrs]
    db_coords = [a['db_coords'] for a in attrs]

    # Get local ndarrays.
    db_ndarrays = array.get_ndarrays()

    # When preparing examples for the protocol release, we need to
    # adjust the version number manually. Otherwise this would be left alone.
    manual_version_update = True
    if manual_version_update:
        manual_version = '0.10.0'
        db_version = [manual_version for version in db_version]

    print("%s" % (title))
    print("%s" % ('`' * len(title)))
    print()
    print(text)
    print()

    # Global array plot.
    if global_plot_filename is not None:
        rst_plot(global_plot_filename)

    # Full (undistributed) array:
    full_array = array.toarray()
    print("The full (undistributed) array:")
    lines = [">>> full_array"] + rst_lines(full_array)
    code_lines = rst_code(lines)
    rst_print_lines([code_lines])

    # Properties that are the same on all processes:
    print("In all processes, we have:")
    lines = []
    lines += [">>> distbuffer = local_array.__distarray__()"]
    lines += [">>> distbuffer.keys()"] + rst_lines(db_keys[0])
    lines += [">>> distbuffer['__version__']"] + rst_lines(db_version[0])
    code_lines = rst_code(lines)
    rst_print_lines([code_lines])

    # Local arrays / properties that vary per engine.
    print("The local arrays, on each separate engine:")
    print()

    # Local array plot.
    if local_plot_filename is not None:
        rst_plot(local_plot_filename)

    # Properties that change per-process:
    lines_list = []
    for rank, (keys, version, buffer, dim_data, ndarray, coord) in enumerate(
            zip(db_keys,
                db_version,
                db_buffer,
                db_dim_data,
                db_ndarrays,
                db_coords)):
        # Skip if local ndarray is empty, as there is no local plot.
        if ndarray.size == 0:
            continue
        header = "In process %r:" % (coord,)
        lines = []
        lines += [">>> distbuffer['buffer']"] + rst_lines(buffer)
        lines += [">>> distbuffer['dim_data']"] + rst_lines(dim_data)
        code_lines = rst_code(lines)
        lines = [header] + code_lines
        lines_list.append(lines)
    # Print as table with nice layout.
    num_local_properties = len(lines_list)
    if (num_local_properties % 2) == 0:
        # 2 X N grid
        rows, cols = (2, num_local_properties // 2)
    else:
        # N x 1 grid
        rows, cols = (num_local_properties, 1)
    rst_table(rows, cols, lines_list)


def create_distribution_plot_and_documentation(context, params):
    """Create an array distribution plot and the related .rst documentation."""

    def shape_text(shape):
        """ Get a text string describing the array shape. """
        # Always want to display at least N X M.
        if len(shape) == 1:
            shape = (1, shape[0])
        shape_labels = ['%d' % (s) for s in shape]
        shape_text = ' X '.join(shape_labels)
        return shape_text

    title = params['title']
    labels = params['labels']
    shape = params['shape']
    grid_shape = params.get('grid_shape', None)
    text = params.get('text', None)
    dist = params.get('dist', None)
    dimdata = params.get('dimdata', None)
    filename = params.get('filename', None)
    skip = params.get('skip', False)

    if skip:
        return

    # Create array, either from dist or dimdata.
    if dist is not None:
        distribution = Distribution(context, shape, dist=dist,
                                    grid_shape=grid_shape)
    elif dimdata is not None:
        distribution = Distribution.from_global_dim_data(context, dimdata)
    else:
        raise ValueError('Must provide either dist or dimdata.')
    array = context.empty(distribution)

    # Fill the array. This is slow but not a real problem here.
    value = 0.0
    if len(shape) == 1:
        for i in range(shape[0]):
            array[i] = value
            value += 1.0
    elif len(shape) == 2:
        for row in range(shape[0]):
            for col in range(shape[1]):
                array[row, col] = value
                value += 1.0
    elif len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    array[i, j, k] = value
                    value += 1.0
    else:
        # TODO: Even better would be to generalize this to any dimensions.
        raise ValueError('Array must be 1, 2, or 3 dimensional.')

    # Get all process grid coordinates.
    # This is duplicating work in print_array_documentation(),
    # but it is needed for the local array plots.
    def _get_process_coords(local_arr):
        return local_arr.cart_coords
    process_coords = context.apply(_get_process_coords,
                                   (array.key,),
                                   targets=array.targets)

    # Plot title and axis labels.
    plot_title = title + ' ' + shape_text(shape) + '\n'
    if len(shape) == 1:
        # add more space for cramped plot.
        plot_title += '\n'
    xlabel = 'Axis 1, %s' % (labels[1])
    ylabel = 'Axis 0, %s' % (labels[0])

    # Documentation title and text description.
    doc_title = title
    dist_text = ' X '.join(["'%s'" % (label) for label in labels])
    # Choose 'a' vs 'an' appropriately.
    if title[0] in 'aeiouAEIOU':
        article = 'an'
    else:
        article = 'a'
    doc_text = 'A (%s) array, with %s %s (%s) distribution over a (%s) process grid.' % (
        shape_text(shape), article, title, dist_text, shape_text(array.grid_shape))
    if text is not None:
        doc_text = doc_text + "\n\n" + text

    # Filenames for array plots.
    global_plot_filename = filename
    local_plot_filename = None
    if global_plot_filename is not None:
        root, ext = os.path.splitext(global_plot_filename)
        local_plot_filename = root + '_local' + ext

    # Create plot.
    if len(shape) in [1, 2]:
        plotting.plot_array_distribution(
            array,
            process_coords,
            title=plot_title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=True,
            global_plot_filename=global_plot_filename,
            local_plot_filename=local_plot_filename)
    else:
        # Not plottable, avoid writing links to missing plots.
        global_plot_filename = None
        local_plot_filename = None

    # Print documentation.
    print_array_documentation(
        context,
        array,
        title=doc_title,
        text=doc_text,
        global_plot_filename=global_plot_filename,
        local_plot_filename=local_plot_filename)


def create_distribution_plot_and_documentation_all(context):
    """Create plots for the distributed array protocol documentation."""
    # Some random values for undistributed example.
    # Use a fixed seed for reproducibility.
    rows, cols = 5, 9
    seed(0x12345670)
    row_indices = permutation(range(rows))
    col_indices = permutation(range(cols))

    #
    # Examples intended for 3 processes:
    #
    params_list_3 = [
        # Same results as old 'b', 'n'.
        {'shape': (5, 9),
         'grid_shape': (3, 1),
         'title': 'Block, Block',
         'labels': ('b', 'b'),
         'filename': 'images/plot_block_block_3x1.png',
         'dimdata': (
             {'dist_type': 'b', 'bounds': [0, 2, 4, 5]},
             {'dist_type': 'b', 'bounds': [0, 9]},
          ),
        },
        # Same results as old 'n', 'b'.
        {'shape': (5, 9),
         'grid_shape': (1, 3),
         'title': 'Block, Block',
         'labels': ('b', 'b'),
         'filename': 'images/plot_block_block_1x3.png',
         'dimdata': (
             {'dist_type': 'b', 'bounds': [0, 5]},
             {'dist_type': 'b', 'bounds': [0, 3, 6, 9]},
          ),
        },
    ]

    #
    # Examples intended for 4 processes:
    #
    params_list_4 = [
        # Some simple description examples.
        {'shape': (5, 9),
         'grid_shape': (2, 2),
         'title': 'Block, Block',
         'labels': ('b', 'b'),
         'filename': 'images/plot_block_block_2x2.png',
         'dist': ('b', 'b'),
        },
        {'shape': (5, 9),
         'grid_shape': (2, 2),
         'title': 'Block, Cyclic',
         'labels': ('b', 'c'),
         'filename': 'images/plot_block_cyclic.png',
         'dist': ('b', 'c'),
        },
        {'shape': (5, 9),
         'grid_shape': (2, 2),
         'title': 'Cyclic, Cyclic',
         'labels': ('c', 'c'),
         'filename': 'images/plot_cyclic_cyclic.png',
         'dist': ('c', 'c'),
        },
        # irregular-block, irregular-block
        {'shape': (5, 9),
         'title': 'Irregular-Block, Irregular-Block',
         'labels': ('b', 'b'),
         'filename': 'images/plot_irregularblock_irregularblock.png',
         'dimdata': (
             {'dist_type': 'b', 'bounds': [0, 1, 5]},
             {'dist_type': 'b', 'bounds': [0, 2, 9]},
          ),
        },
        # blockcyclic-blockcyclic: Like cyclic-cyclic but with block_size=2.
        {'shape': (5, 9),
         'title': 'Block-Cyclic, Block-Cyclic',
         'labels': ('c', 'c'),
         'filename': 'images/plot_blockcyclic_blockcyclic.png',
         'dimdata': (
             {'size': 5, 'dist_type': 'c', 'proc_grid_size': 2, 'block_size': 2},
             {'size': 9, 'dist_type': 'c', 'proc_grid_size': 2, 'block_size': 2},
          ),
        },
        # block-padded, block-padded: Block with padding = (1, 1).
        {'shape': (5, 9),
         'title': 'Block-Padded, Block-Padded',
         'labels': ('b', 'b'),
         'filename': 'images/plot_blockpad_blockpad.png',
         # The padding is not actually used yet, so this is not a meaningful
         # example now.
         'skip': True,
         'dimdata': (
             {'dist_type': 'b', 'bounds': [0, 2, 5], 'boundary_padding': 1},
             {'dist_type': 'b', 'bounds': [0, 4, 9], 'boundary_padding': 1},
          ),
        },
        # 1D unstructured example. Skipped for now but may be a useful example.
        {'shape': (40,),
         'skip': True,
         'title': 'Unstructured',
         'labels': ('u', 'u'),
         'filename': 'images/plot_unstructured.png',
         'dimdata': (
             {'dist_type': 'u',
              'indices': [
                   [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
                   [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
                   [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
                   [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
               ], },
          ),
        },
        # Unstructured, unstructured.
        {'shape': (rows, cols),
         'title': 'Unstructured, Unstructured',
         'labels': ('u', 'u'),
         'filename': 'images/plot_unstruct_unstruct.png',
         'dimdata': (
             {'dist_type': 'u',
              'indices': [row_indices[:rows // 2], row_indices[rows // 2:]],
             },
             {'dist_type': 'u',
              'indices': [col_indices[:cols // 2], col_indices[cols // 2:]],
             },
          ),
        },
    ]

    #
    # Examples intended for 8 processes:
    #
    params_list_8 = [
        # A 3D array.
        {
         'shape': (5, 9, 3),
         'grid_shape': (2, 2, 2),
         'title': 'Cyclic, Block, Cyclic',
         'labels': ('c', 'b', 'c'),
         'filename': 'images/plot_cyclic_block_cyclic.png',
         'dist': ('c', 'b', 'c'),
        },
    ]

    # Get the examples to use for the number of engines.
    param_list = []
    num_engines = len(context.targets)
    if num_engines == 3:
        # Examples that only use 3 processes.
        param_list.extend(params_list_3)
    elif num_engines == 4:
        # 1,2 dimension cases with 4 engines give nicer plots.
        param_list.extend(params_list_4)
    elif num_engines == 8:
        # 3 dimension cases require 8 engines for now.
        param_list.extend(params_list_8)
    else:
        # No examples for this engine count.
        pass

    # Crunch...
    for params in param_list:
        create_distribution_plot_and_documentation(context, params)


def main():
    context = distarray.globalapi.Context()
    num_targets = len(context.targets)
    # Examples are designed for various engine counts...
    engine_counts = [3, 4, 8]
    need_targets = max(engine_counts)
    if num_targets < need_targets:
        raise ValueError('Need at least %d engines for all the examples, '
                         'but only %d are available.' % (need_targets, num_targets))
    for engine_count in engine_counts:
        context_n = distarray.globalapi.Context(targets=range(engine_count))
        create_distribution_plot_and_documentation_all(context_n)


if __name__ == '__main__':
    import argparse
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.parse_args()
    main()
