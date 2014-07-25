# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Plot the results of the Julia set timings.
"""

from __future__ import print_function, division

import sys
import random
import json

import numpy
import pandas as pd
import matplotlib
from matplotlib import pyplot

from distarray.externals.six import next


# tweak plot styling

CBcdict = {
    'Bl': (0, 0, 0),
    'Or': (.9, .6, 0),
    'SB': (.35, .7, .9),
    'bG': (0, .6, .5),
    'Ye': (.95, .9, .25),
    'Bu': (0, .45, .7),
    'Ve': (.8, .4, 0),
    'rP': (.8, .6, .7),
}

matplotlib.rcParams['axes.color_cycle'] = [CBcdict[c] for c in
                                           sorted(CBcdict.keys())]
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 13
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['grid.alpha'] = 0.5


STYLES = ('o-', 'x-', 'v-', '*-', 's-', 'd-')


def read_results(filename):
    """Read the Julia Set timing results from the file.

    Parse into a pandas dataframe.
    """
    def date_parser(c):
        return pd.to_datetime(c, unit='s')
    df = pd.read_json(filename, convert_dates=['Start', 'End'], date_unit='s')
    return df


def process_data(df, ideal_dist='b-n', aggfunc=numpy.min):
    """Process the timing results.

    Add an ideal scaling line from the origin through the first 'b-b' point.
    """
    pdf = df.pivot_table(index='Engines', columns='Dist', values='t_DistArray',
                         aggfunc=aggfunc)
    resolution = df.Resolution.iloc[0]
    kpoints_df = (resolution**2 / pdf) / 1000
    kpoints_df['ideal ' + ideal_dist] = kpoints_df[ideal_dist].iloc[0] * kpoints_df.index.values

    return kpoints_df


def plot_points(df, note, ideal_dist='b-n', aggfunc=numpy.min):
    """Plot the timing results.

    Plot an ideal scaling line from the origin through the first 'b-b' point.
    """
    styles = iter(STYLES)
    for col in df.columns:
        if col.startswith("ideal"):
            continue
        else:
            pyplot.plot(df.index, df[col], next(styles))
    pyplot.plot(df.index, df['ideal ' + ideal_dist], '--')

    pyplot.suptitle(note[0])
    pyplot.title(note[2], fontsize=12)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('kpoints / s')
    pyplot.legend(df.columns, loc='upper left')
    pyplot.grid(axis='y')
    pyplot.show()


def main(filename):
    # Read and parse timing results.
    df, note = read_results(filename)
    pdf = process_data(df)
    plot_points(pdf, note)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage = 'Usage: python plot_results.py <results filename>'
        print(usage)
        exit(1)
    filename = sys.argv[1]
    main(filename)
