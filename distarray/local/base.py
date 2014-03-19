# encoding: utf-8
#----------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
#----------------------------------------------------------------------------


def distribute_block_indices(dd):
    """Fill in `start` and `stop` in dimdict `dd`."""
    if ('start' in dd) and ('stop' in dd):
        return

    nelements = dd['size'] // dd['proc_grid_size']
    if dd['size'] % dd['proc_grid_size'] != 0:
        nelements += 1

    dd['start'] = dd['proc_grid_rank'] * nelements
    if dd['start'] > dd['size']:
        dd['start'] = dd['size']
        dd['stop'] = dd['size']

    dd['stop'] = dd['start'] + nelements
    if dd['stop'] > dd['size']:
        dd['stop'] = dd['size']


def distribute_cyclic_indices(dd):
    """Fill in `start` given dimdict `dd`."""
    if 'start' in dd:
        return
    else:
        dd['start'] = dd['proc_grid_rank']


def distribute_indices(dim_data):
    """Fill in missing index related keys...

    for supported dist_types.
    """
    distribute_fn = {
        'n': lambda dd: None,
        'b': distribute_block_indices,
        'c': distribute_cyclic_indices,
        'u': lambda dd: None,
    }
    for dim in dim_data:
        distribute_fn[dim['dist_type']](dim)


def arecompatible(a, b):
    """Do these arrays have the same compatibility hash?"""
    return a.compatibility_hash() == b.compatibility_hash()
