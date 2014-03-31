# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plot distributions for some distarrays for the protocol documentation.

The output from this example program should be directly usable in the
distributed array protocol documentation, if saved as examples.rst.
The output .png files should be copied to the images folder as well.
"""

from __future__ import print_function

import os.path
from pprint import pformat
from numpy.random import permutation, seed

import distarray
from distarray import plotting


def print_array_documentation(context,
                              array,
                              title,
                              text,
                              global_plot_filename,
                              local_plot_filename,
                              verbose=False):
    """ Print some properties of the array.

    The output is rst formatted, so that it can be directly
    used for documentation. It includes a plot of the array distribution,
    some properties that are same for each process, and properties that
    vary per process.

    Parameters
    ----------
    context : distarray.Context
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
    verbose : bool
        If True, some extra information is printed, that would
        be overly verbose for the real protocol documentation.
    """

    def rst_print(obj):
        """ Return text that formats obj nicely for an .rst document. """
        text = pformat(obj)
        # pformat() gives blank lines for 3d arrays, which confuse Sphinx.
        lines = text.split('\n')
        trims = [line for line in lines if len(line) > 0]
        trimmed = '\n'.join(trims)
        return trimmed

    def rst_plot(filename):
        """ Include a plot in the .rst document. """
        print(".. image:: ../images/%s" % (filename))
        # align right does not work as I want.
        #print("   :align: right")
        print()

    # Examine the array on all the engines.
    cmd = 'distbuffer = %s.__distarray__()' % (array.key)
    context._execute(cmd)
    cmd = 'db_keys = distbuffer.keys()'
    context._execute(cmd)
    cmd = 'db_version = distbuffer["__version__"]'
    context._execute(cmd)
    cmd = 'db_buffer = distbuffer["buffer"]'
    context._execute(cmd)
    cmd = 'db_dim_data = distbuffer["dim_data"]'
    context._execute(cmd)
    # Get data from each engine.
    db_keys = context.view['db_keys']
    db_version = context.view['db_version']
    db_buffer = context.view['db_buffer']
    db_dim_data = context.view['db_dim_data']

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
    print()
    print(">>> full_array")
    print(rst_print(full_array))
    print()

    # Result of get_localarrays() and get_ndarrays().
    # This is mainly for debugging and will eventually not be needed.
    if verbose:
        local_arrays = array.get_localarrays()
        nd_arrays = array.get_ndarrays()
        print("Result of get_localarrays() and get_ndarrays():")
        print()
        print(">>> get_localarrays()")
        print(rst_print(local_arrays))
        print(">>> get_ndarrays()")
        print(rst_print(nd_arrays))
        print()

    # Properties that are the same on all processes:
    print("In all processes:")
    print()
    print(">>> distbuffer = local_array.__distarray__()")
    print(">>> distbuffer.keys()")
    print(rst_print(db_keys[0]))
    print(">>> distbuffer['__version__']")
    print(rst_print(db_version[0]))
    print()

    # Local array plot.
    if local_plot_filename is not None:
        rst_plot(local_plot_filename)

    # Properties that change per-process:
    for p, (keys, version, buffer, dim_data) in enumerate(
            zip(db_keys, db_version, db_buffer, db_dim_data)):
        print("In process %d:" % (p))
        print()
        print(">>> distbuffer['buffer']")
        print(rst_print(buffer))
        print(">>> distbuffer['dim_data']")
        print(rst_print(dim_data))
        print()


def create_distribution_plot_and_documentation(context, params):
    """ Create an array distribution plot,
    and the related .rst documentation.
    """

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
    text = params.get('text', None)
    dist = params.get('dist', None)
    dimdata = params.get('dimdata', None)
    filename = params.get('filename', None)
    skip = params.get('skip', False)

    if skip:
        return

    # Skip if dimdata count does not match context.
    if dimdata is not None:
        num_dimdata = len(dimdata)
        num_targets = len(context.targets)
        if num_dimdata != num_targets:
            return

    # Create array, either from dist or dimdata.
    if dist is not None:
        array = context.empty(shape, dist=dist)
    elif dimdata is not None:
        array = context.from_dim_data(dimdata)
    else:
        raise ValueError('Must provide either dist or dimdata.')

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

    # Plot title and axis labels.
    plot_title = title + ' ' + shape_text(shape) + '\n'
    if len(shape) == 1:
        # add more space for cramped plot.
        plot_title += '\n'
    xlabel = 'Axis 1, %s' % (labels[1])
    ylabel = 'Axis 0, %s' % (labels[0])

    # Documentation title and text description.
    doc_title = title
    dist_text = ' x '.join(["'%s'" % (label) for label in labels])
    doc_text = 'A (%s) array, with a %s (%s) distribution over a (%s) process grid.' % (
        shape_text(shape), title, dist_text, shape_text(array.grid_shape))
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


def create_distribution_plot_and_documentation_all(
    context, dimlist, add_header=False):
    """ Create plots for the distributed array protocol documentation. """

    # Some random values for undistributed example.
    # Use a fixed seed for reproducibility.
    rows, cols = 5, 9
    seed(0x12345670)
    row_indices = permutation(range(rows))
    col_indices = permutation(range(cols))

    params_list = [
        # Examples using simple dist specification.
        {'shape': (5, 9),
         'title': 'Block, Nondistributed',
         'labels': ('b', 'n'),
         'filename': 'plot_block_nondist.png',
         'dist': ('b', 'n'),
        },
        {'shape': (5, 9),
         'title': 'Nondistributed, Block',
         'labels': ('n', 'b'),
         'filename': 'plot_nondist_block.png',
         'dist': ('n', 'b'),
        },
        {'shape': (5, 9),
         'title': 'Block, Block',
         'labels': ('b', 'b'),
         'filename': 'plot_block_block.png',
         'dist': ('b', 'b'),
        },
        {'shape': (5, 9),
         'title': 'Block, Cyclic',
         'labels': ('b', 'c'),
         'filename': 'plot_block_cyclic.png',
         'dist': ('b', 'c'),
        },
        {'shape': (5, 9),
         'title': 'Cyclic, Cyclic',
         'labels': ('c', 'c'),
         'filename': 'plot_cyclic_cyclic.png',
         'dist': ('c', 'c'),
        },
        # A 3D array. This needs more than 4 engines.
        {
         'shape': (5, 9, 3),
         #'shape': (4, 4, 4),
         'title': 'Cyclic, Block, Cyclic',
         'labels': ('c', 'b', 'c'),
         'filename': 'plot_cyclic_block_cyclic.png',
         'dist': ('c', 'b', 'c'),
        },
        # regular-block, irregular-block
        {'shape': (5, 9),
         'title': 'Block, Irregular-Block',
         'labels': ('b', 'b'),
         'filename': 'plot_block_irregularblock.png',
         'skip': True,    # IndexErrors now???
         'dimdata': [
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'start': 0,
              'stop': 2},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'start': 2,
              'stop': 6},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'start': 6,
              'stop': 7},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'start': 7,
              'stop': 9},
             ),
          ],
        },
        # Examples requiring custom dimdata.
        # blockcyclic-blockcyclic: Like cyclic-cyclic but with block_size=2.
        {'shape': (5, 9),
         'title': 'BlockCyclic, BlockCyclic',
         'labels': ('bc', 'bc'),
         'filename': 'plot_blockcyclic_blockcyclic.png',
         'skip': True,    # IndexErrors now???
         'dimdata': [
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 5,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 9,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 5,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 9,
              'start': 2}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 5,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 9,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 5,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 9,
              'start': 2}),
          ],
        },
        # block-padded, block-padded: Block with padding = (1, 1).
        {'shape': (5, 9),
         'title': 'BlockPadded, BlockPadded',
         'labels': ('bp', 'bp'),
         'filename': 'plot_blockpad_blockpad.png',
         'skip': True,    # IndexErrors now???
         'dimdata': [
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'start': 0,
              'stop': 2,
              'padding': (1, 1)},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'start': 0,
              'stop': 4,
              'padding': (1, 1)},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'start': 0,
              'stop': 2,
              'padding': (1, 1)},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'start': 4,
              'stop': 9,
              'padding': (1, 1)},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'start': 2,
              'stop': 5,
              'padding': (1, 1)},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'start': 0,
              'stop': 4,
              'padding': (1, 1)},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'start': 2,
              'stop': 5,
              'padding': (1, 1)},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'start': 4,
              'stop': 9,
              'padding': (1, 1)},
             ),
          ],
        },
        # 1D unstructured example. Skipped for now but may be a useful example.
        {'shape': (40,),
         'skip': True,
         'title': 'Unstructured',
         'labels': ('u', 'u'),
         'filename': 'plot_unstructured.png',
         'dimdata': [
            ({'dist_type': 'u',
              'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'size': 40},),
            ({'dist_type': 'u',
              'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'size': 40},),
            ({'dist_type': 'u',
              'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'size': 40},),
            ({'dist_type': 'u',
              'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'size': 40},)],
        },
        # Unstructured, unstructured.
        {'shape': (rows, cols),
         'title': 'Unstructured, Unstructured',
         'labels': ('u', 'u'),
         'filename': 'plot_unstruct_unstruct.png',
         'dimdata': [
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows // 2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols // 2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[:rows // 2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols // 2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows // 2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[:cols // 2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': cols},
             ),
             (
              {'dist_type': 'u',
               'indices': row_indices[rows // 2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': rows},
              {'dist_type': 'u',
               'indices': col_indices[cols // 2:],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': cols},
             )],
        },
    ]

    test_params_list = [
        {'shape': (5, 9),
         'title': 'Block, Irregular-Block',
         'labels': ('b', 'b'),
         'filename': 'plot_block_irregularblock.png',
         'dimdata': [
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'start': 0,
              'stop': 2},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'start': 2,
              'stop': 6},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'start': 6,
              'stop': 7},
             ),
            (
             {'size': 5,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 1,
              'start': 0,
              'stop': 5},
             {'size': 9,
              'dist_type': 'b',
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'start': 7,
              'stop': 9},
             ),
          ],
        },
#         {'shape': (5, 9),
#          'title': 'Cyclic, Cyclic',
#          'labels': ('c', 'c'),
#          'filename': 'plot_cyclic_cyclic.png',
#          'dist': ('c', 'c'),
#         },
#         {'shape': (5, 9),
#          'title': 'Block, Nondistributed',
#          'labels': ('b', 'n'),
#          'filename': 'plot_block_nondist.png',
#          'dist': ('b', 'n'),
#         },
#         {'shape': (5, 9),
#          'title': 'Block, Cyclic',
#          'labels': ('b', 'c'),
#          'filename': 'plot_block_cyclic.png',
#          'dist': ('b', 'c'),
#         },
    ]

    # Document section header
    if add_header:
        print('Automatically Generated Examples')
        print('--------------------------------')
        print()

    for params in params_list:
    #for params in test_params_list:
        num_dims = len(params['shape'])
        if num_dims in dimlist:
            create_distribution_plot_and_documentation(context, params)


def main():
    context = distarray.Context()
    num_targets = len(context.targets)
    # The 1,2-D examples work best with a 4-process grid, and
    # the 3D examples work best with an 8-process grid.
    # This is clumsy, I don't like it, but should be ok for now.
    if num_targets == 4:
        dimlist = [1, 2]
    elif num_targets == 8:
        dimlist = [3]
        dimlist = [1, 2, 3]
    else:
        # Try them all, but GridShapeErrors are likely!
        dimlist = [1, 2, 3]
    create_distribution_plot_and_documentation_all(context, dimlist)


if __name__ == '__main__':
    main()
