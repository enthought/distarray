# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------

"""
Plot distributions for some distarrays for the protocol documentation.
"""

import matplotlib
import distarray
from distarray import plotting


# Use 2 processors to match examples.
#context = distarray.Context(targets=[0, 1])

# All 4 for less GridShapeError.
context = distarray.Context()


def plot_distribution(a, title, filename, interactive=True):
    plotting.plot_array_distribution_2d(a, draw_legend=True)
    if title is not None:
        matplotlib.pyplot.title(title)
    if filename is not None:
        matplotlib.pyplot.savefig(filename)
    if interactive:
        matplotlib.pyplot.show()


def create_distribution_plot(params):
    """ Create an array distribution plot,
    suitable for the protocol documentation. """
    shape, dist = params['shape'], params['dist']
    array = context.zeros(shape, dist=dist)
    # Add newline to title for better spacing.
    title, filename = params['title'], params['filename']
    # Create a nice title.
    full_title = title + ' %d-by-%d' % (shape[0], shape[1]) + '\n'
    plot_distribution(array, full_title, filename, False)


def create_distributed_protocol_documentation_plots():
    """ Create plots for the distributed array protocol documentation. """ 
    params_list = [
        {'shape': (2, 10),
         'dist': ('b', 'n'),
         'title': 'Block-Nondistributed',
         'filename': 'plot_block_nondist.png',
        },
        {'shape': (2, 10),
         'dist': ('n', 'b'),
         'title': 'Nondistributed-Block',
         'filename': 'plot_nondist_block.png',
        },
        {'shape': (10, 10),
         'dist': ('b', 'b'),
         'title': 'Block-Block',
         'filename': 'plot_block_block.png',
        },
        {'shape': (10, 10),
         'dist': ('b', 'c'),
         'title': 'Block-Cyclic',
         'filename': 'plot_block_cyclic.png',
        },
        {'shape': (10, 10),
         'dist': ('c', 'c'),
         'title': 'Cyclic-Cyclic',
         'filename': 'plot_cyclic_cyclic.png',
        },
#         {'shape': (),
#          'dist': (),
#          'title': '',
#          'filename': 'plot_.png',
#         },
    ]
    for params in params_list:
        create_distribution_plot(params)


if __name__ == '__main__':
    create_distributed_protocol_documentation_plots()

    if False:
        # Current working test...
        a = context.zeros((10, 10), dist=('u', 'u'))
        plot_distribution(a, 'Unstructured-Unstructured\n', 'plot_unstruct_unstruct.png')
