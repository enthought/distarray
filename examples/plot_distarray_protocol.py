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
    plotting.plot_array_distribution_2d(a, draw_legend=True, xlabel=xlabel, ylabel=ylabel)
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
    # Get the full array.
    full_array = array.toarray()

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
    print("The full (undistributed) array:")
    print()
    print(">>> full_array")
    pprint(full_array)
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
    shape, dist = params['shape'], params['dist']
    array = context.empty(shape, dist=dist)
    # Fill the array [slow].
    value = 1.0
    for row in range(shape[0]):
        for col in range(shape[1]):
            array[row, col] = value
            value += 1.0
    # Add newline to title for better spacing.
    title, filename = params['title'], params['filename']
    # Create a nice title.
    full_title = title + ' %d-by-%d' % (shape[0], shape[1]) + '\n'
    # Nice labels for axes.
    xlabel = 'Axis 0, %s' % (dist[0])
    ylabel = 'Axis 1, %s' % (dist[1])
    # Text description for documentation.
    # I am not sure how to determine the process grid shape.
    text = '%d X %d array, %s distribution, distributed over a 2 X 2 process grid.' % (
        shape[0], shape[1], title)
    if 'text' in params:
        text = text + "\n\n" + params['text']
    plot_distribution(array, full_title, xlabel, ylabel, filename, False)
    # Print properties on engines.
    print_engine_array(context, array, title, text, filename)


def create_distributed_protocol_documentation_plots():
    """ Create plots for the distributed array protocol documentation. """ 
    params_list = [
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
    ]

    # Document section header
    print('Automatically Generated Examples')
    print('--------------------------------')
    print()

    for params in params_list:
    #for params in [params_list[0]]:
    #for params in [params_list[4]]:
        create_distribution_plot(params)


from numpy.testing import assert_allclose


def test_from_dim_data():
    total_size = 40
    ddpp = [
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
          'size': 40},)]
    distarr = context.from_dim_data(ddpp)
    for i in range(total_size):
        distarr[i] = i
    localarrays = distarr.get_localarrays()
    for i, arr in enumerate(localarrays):
        assert_allclose(arr, ddpp[i][0]['indices'])


if __name__ == '__main__':
    if True:
        test_from_dim_data()
    
    if True:
        create_distributed_protocol_documentation_plots()

    if False:
        # Current working test...
        a = context.zeros((10, 10), dist=('u', 'u'))
        plot_distribution(a, 'Unstructured-Unstructured\n', 'plot_unstruct_unstruct.png')
