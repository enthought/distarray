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

import matplotlib
from pprint import pprint

import distarray
from distarray import plotting

# Use 2 processors to match examples.
#context = distarray.Context(targets=[0, 1])

# All 4 for less GridShapeError.
context = distarray.Context()


def plot_distribution(a, title, xlabel, ylabel, filename, interactive=True):
    plotting.plot_array_distribution(
        a, xlabel=xlabel, ylabel=ylabel, legend=True)
    if title is not None:
        matplotlib.pyplot.title(title)
    if filename is not None:
        matplotlib.pyplot.savefig(filename)
    if interactive:
        matplotlib.pyplot.show()


def print_engine_array(context, array, title, text, filename):
    """ Print some properties of the array on each engine.

    This is formatted to fit nicely into the documentation.
    """
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

    # Add image.
    print(".. image:: ../images/%s" % (filename))
    # align right does not work as I want.
    #print("   :align: right")
    print()

    # Full (undistributed) array:
    full_array = array.toarray()
    print("The full (undistributed) array:")
    print()
    print(">>> full_array")
    pprint(full_array)
    print()

    # Result of get_localarrays() and get_ndarrays().
    # This is mainly for debugging and will eventually not be needed.
    local_arrays = array.get_localarrays()
    nd_arrays = array.get_ndarrays()
    print("Result of get_localarrays() and get_ndarrays():")
    print()
    print(">>> get_localarrays()")
    pprint(local_arrays)
    print(">>> get_ndarrays()")
    pprint(nd_arrays)
    print()

    # Properties that are the same on all processes:
    print("In all processes:")
    print()
    print(">>> distbuffer = local_array.__distarray__()")
    print(">>> distbuffer.keys()")
    pprint(db_keys[0])
    print(">>> distbuffer['__version__']")
    pprint(db_version[0])
    print()

    # Properties that change per-process:
    for p, (keys, version, buffer, dim_data) in enumerate(
            zip(db_keys, db_version, db_buffer, db_dim_data)):
        print("In process %d:" % (p))
        print()
        print(">>> distbuffer['buffer']")
        pprint(buffer)
        print(">>> distbuffer['dim_data']")
        pprint(dim_data)
        print()


def create_distribution_plot(params):
    """ Create an array distribution plot,
    suitable for the protocol documentation. """

    def shape_text(shape):
        """ Get a text string describing the array shape. """
        shape_labels = ['%d' % (s) for s in shape]
        shape_text = ' X '.join(shape_labels)
        return shape_text

    if 'skip' in params and params['skip']:
        print('Skipping %s...' % (params['title']))
        return

    if 'dimdata' not in params:
        shape, dist = params['shape'], params['dist']
        array = context.empty(shape, dist=dist)
    else:
        shape, dimdata = params['shape'], params['dimdata']
        dist = ('x', 'x')
        array = context.from_dim_data(dimdata)

    # Fill the array [slow].
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
    else:
        raise ValueError('Array must be 1 or 2 dimensional.')

    # Fixup shape
    if len(shape) == 1:
        shape = (1, shape[0])

    # Add newline to title for better spacing.
    title, filename = params['title'], params['filename']
    # Create a nice title.
    full_title = title + ' %d-by-%d' % (shape[0], shape[1]) + '\n'
    # Nice labels for axes.
    xlabel = 'Axis 1, %s' % (dist[1])
    ylabel = 'Axis 0, %s' % (dist[0])
    # Text description for documentation.
    grid_shape = array.grid_shape
    text = '%s array, %s distribution, distributed over %s process grid.' % (
        shape_text(shape), title, shape_text(grid_shape))
    if 'text' in params:
        text = text + "\n\n" + params['text']
    plot_distribution(array, full_title, xlabel, ylabel, filename, False)
    # Print properties on engines.
    print_engine_array(context, array, title, text, filename)


def create_distributed_protocol_documentation_plots():
    """ Create plots for the distributed array protocol documentation. """
    params_list = [
        # Examples using simple dist specification.
        {'shape': (4, 8),
         'dist': ('b', 'n'),
         'title': 'Block, Nondistributed',
         'filename': 'plot_block_nondist.png',
         'text': '''Some description of Block, Nondistributed.''',
        },
        {'shape': (4, 8),
         'dist': ('n', 'b'),
         'title': 'Nondistributed, Block',
         'filename': 'plot_nondist_block.png',
        },
        {'shape': (4, 8),
         'dist': ('b', 'b'),
         'title': 'Block, Block',
         'filename': 'plot_block_block.png',
        },
        {'shape': (4, 8),
         'dist': ('b', 'c'),
         'title': 'Block, Cyclic',
         'filename': 'plot_block_cyclic.png',
        },
        {'shape': (4, 8),
         'dist': ('c', 'c'),
         'title': 'Cyclic, Cyclic',
         'filename': 'plot_cyclic_cyclic.png',
        },
        # Examples requiring custom dimdata.
        # blockcyclic-blockcyclic: Like cyclic-cyclic but with block_size=2.
        {'shape': (4, 8),
         'dimdata': [
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 4,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 8,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 4,
              'start': 0},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 8,
              'start': 2}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 4,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 0,
              'proc_grid_size': 2,
              'size': 8,
              'start': 0}),
            ({'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 4,
              'start': 2},
             {'block_size': 2,
              'dist_type': 'c',
              'proc_grid_rank': 1,
              'proc_grid_size': 2,
              'size': 8,
              'start': 2}),
          ],
         'title': 'BlockCyclic[2], BlockCyclic[2]',
         'filename': 'plot_blockcyclic_blockcyclic.png',
        },
        # 1D unstructured example.
        {'shape': (40,),
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
         'title': 'Unstructured',
         'filename': 'plot_unstructured.png',
        },
        # 1D padded example.
        {'shape': (20,),
         'skip': True,
         'dimdata': [
            ({'size': 20,
              'dist_type': 'b',
              'proc_grid_rank': 0,
              'proc_grid_size': 4,
              'start': 0,
              'stop': 5,
              'padding': (1, 1)},),
            ({'size': 20,
              'dist_type': 'b',
              'proc_grid_rank': 1,
              'proc_grid_size': 4,
              'start': 5,
              'stop': 10,
              'padding': (1, 1)},),
            ({'size': 20,
              'dist_type': 'b',
              'proc_grid_rank': 2,
              'proc_grid_size': 4,
              'start': 10,
              'stop': 15,
              'padding': (1, 1)},),
            ({'size': 20,
              'dist_type': 'b',
              'proc_grid_rank': 3,
              'proc_grid_size': 4,
              'start': 15,
              'stop': 20,
              'padding': (1, 1)},),
          ],
         'title': 'Padded',
         'filename': 'plot_padded.png',
        },
        #
        # Some attempts at a 2D unstructured array.
        # I have not been able to get these to work yet.
        #
        {'shape': (3, 3),
         'skip': True,
         'dimdata': [
             (
              {'dist_type': 'u',
               #'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
               'indices': [0, 1],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 3},
              {'dist_type': 'u',
               #'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
               'indices': [0, 1],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 3},
             ),
             (
              {'dist_type': 'u',
               #'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
               'indices': [2],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 3},
              {'dist_type': 'u',
               #'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
               'indices': [2],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 3},
             ),
             (
              {'dist_type': 'u',
               #'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
               'indices': [],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 3},
              {'dist_type': 'u',
               #'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
               'indices': [],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 3},
             ),
             (
              {'dist_type': 'u',
               #'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
               'indices': [],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 3},
              {'dist_type': 'u',
               #'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
               'indices': [],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 3},
             )],
         'title': 'Attempt #1: Unstructured, Unstructured',
         'filename': 'plot_unstruct_unstruct_1.png',
        },
        {'shape': (40, 40),
         'skip': True,
         'dimdata': [
             (
              {'dist_type': 'u',
               'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 40},
              {'dist_type': 'u',
               'indices': [29, 38, 18, 19, 11, 33, 10, 1, 22, 25],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 40},
             ),
             (
              {'dist_type': 'u',
               'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 40},
              {'dist_type': 'u',
               'indices': [5, 15, 34, 12, 16, 24, 23, 39, 6, 36],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 40},
             ),
             (
              {'dist_type': 'u',
               'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 40},
              {'dist_type': 'u',
               'indices': [0, 7, 27, 4, 32, 37, 21, 26, 9, 17],
               'proc_grid_rank': 0,
               'proc_grid_size': 2,
               'size': 40},
             ),
             (
              {'dist_type': 'u',
               'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 40},
              {'dist_type': 'u',
               'indices': [35, 14, 20, 13, 3, 30, 2, 8, 28, 31],
               'proc_grid_rank': 1,
               'proc_grid_size': 2,
               'size': 40},
             )],
         'title': 'Attempt #2: Unstructured, Unstructured',
         'filename': 'plot_unstruct_unstruct_2.png',
        },
    ]

    # Document section header
    print('Automatically Generated Examples')
    print('--------------------------------')
    print()

    for params in params_list:
    #for params in [params_list[0]]:
    #for params in [params_list[-1]]:
        create_distribution_plot(params)


if __name__ == '__main__':
    create_distributed_protocol_documentation_plots()
