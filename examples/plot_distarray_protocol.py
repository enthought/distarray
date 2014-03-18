# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plot distributions for some distarrays for the protocol documentation.
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


def print_engine_array(context, array, title, filename):
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

    # Print
    print("%s" % (title))
    print("%s" % ('`' * len(title)))
    print("")
    print("Engine properties for: %s" % (title))
    print("")
    for p, (keys, version, buffer, dim_data) in enumerate(
            zip(db_keys, db_version, db_buffer, db_dim_data)):
        print("In process %d:" % (p))
        print("")
        print(">>> distbuffer = a%d.__distarray__()" % (p))
        print(">>> distbuffer.keys()")
        pprint(keys)
        print(">>> distbuffer['__version__']")
        pprint(version)
        print(">>> distbuffer['buffer']")
        pprint(buffer)
        print(">>> distbuffer['dim_data']")
        pprint(dim_data)
        print('')
    # Link to image
    print(".. image:: ../images/%s" % (filename))
    print("")
    print("")


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
    xlabel = 'Processor 0.%s' % (dist[0])
    ylabel = 'Processor 1.%s' % (dist[1])
    plot_distribution(array, full_title, xlabel, ylabel, filename, False)
    # Print properties on engines.
    print_engine_array(context, array, title, filename)


def create_distributed_protocol_documentation_plots():
    """ Create plots for the distributed array protocol documentation. """ 
    params_list = [
        {'shape': (4, 8),
         'dist': ('b', 'n'),
         'title': 'Block, Nondistributed',
         'filename': 'plot_block_nondist.png',
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
    #for params in [params_list[4]]:
        create_distribution_plot(params)


if __name__ == '__main__':
    if True:
        create_distributed_protocol_documentation_plots()

    if False:
        # Current working test...
        a = context.zeros((10, 10), dist=('u', 'u'))
        plot_distribution(a, 'Unstructured-Unstructured\n', 'plot_unstruct_unstruct.png')
