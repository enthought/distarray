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
#    'Ye': (.95, .9, .25),
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


def process_data(df, ideal_dist='numpy', aggfunc=numpy.min):
    """Process the timing results.

    Add an ideal scaling line from the origin through the first 'b-b' point.
    """
    df['Runtime (s)'] = (df['End'] - df['Start']) / numpy.timedelta64(1, 's')
    pdf = df.pivot_table(index='Engines', columns='Dist', values='Runtime (s)',
                         aggfunc=aggfunc)
    resolution = df.groupby(['Engines'])['Resolution'].min()
    mpoints_df = (resolution[:,None]**2 / pdf) / 10**6
    mpoints_df['ideal ' + ideal_dist] = mpoints_df[ideal_dist].iloc[0] * mpoints_df.index.values
    del mpoints_df[ideal_dist]

    return mpoints_df


def plot_points(dfmed, dfmin, dfmax, subtitle, ideal_dist='numpy'):
    """Plot the timing results.

    Plot an ideal scaling line from the origin through the first `ideal_idst`
    point.
    """
    styles = iter(STYLES)
    for col in dfmed.columns:
        if col == 'ideal ' + ideal_dist:
            fmt='--'
        else:
            fmt = next(styles)
        pyplot.errorbar(x=dfmed.index, y=dfmed[col],
                        yerr=[dfmin[col], dfmax[col]],
                        lolims=True, uplims=True,
                        fmt=fmt)

    xmin = dfmed.index.min() - 1
    xmax = dfmed.index.max() + 1
    pyplot.xlim((xmin, xmax))
    pyplot.suptitle("Julia Set Benchmark")
    pyplot.title(subtitle, fontsize=12)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('mpoints / s')
    pyplot.legend(dfmed.columns, loc='upper left')
    pyplot.grid(axis='y')
    pyplot.show()


def plot_ratios(dfmed, dfmin, dfmax, subtitle, ideal_dist='numpy'):
    """Plot the timing results.

    Plot each line divided by ideal.
    """
    ideal = 'ideal ' + ideal_dist
    ratios = []

    for df in (dfmed, dfmin, dfmax):
        if df is None:
            ratios.append(None)
            continue
        ratio = df.div(df[ideal], axis='index')
        del ratio[ideal]
        ratios.append(ratio)
    dfmed, dfmin, dfmax = ratios

    styles = iter(STYLES)
    for col in dfmed.columns:
        fmt = next(styles)
        if dfmin and dfmax:
            yerr = [dfmin[col], dfmax[col]]
        else:
            yerr = None
        pyplot.errorbar(x=dfmed.index, y=dfmed[col],
                        yerr=yerr,
                        lolims=True, uplims=True,
                        fmt=fmt)

    xmin = dfmed.index.min() - 1
    xmax = dfmed.index.max() + 1
    pyplot.xlim((xmin, xmax))
    pyplot.suptitle("Julia Set Benchmark (ratios)")
    pyplot.title(subtitle, fontsize=12)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('mpoints / ideal mpoints')
    pyplot.legend(dfmed.columns, loc='lower left')
    pyplot.grid(axis='y')
    pyplot.show()


def main(filename, ratios=False):
    # Read and parse timing results.
    df = read_results(filename)

    # Build the subtitle
    resolution = df['Resolution'].unique()
    c = df['c'].unique()
    total_time = (df['End'].max() - df['Start'].min()).seconds
    nreps = df.pivot_table(index='Engines', columns='Dist',
                           values='Resolution', aggfunc=len).min().min()

    fmt = "resolution={}, c={}, total_time={:0.2f}s, nreps={}"
    subtitle = fmt.format(resolution, c, total_time, int(nreps))

    pdfmin = process_data(df, aggfunc=numpy.min)
    pdfmax = process_data(df, aggfunc=numpy.max)
    pdfmed = process_data(df, aggfunc=numpy.median)

    if ratios:
        plot_ratios(pdfmed, None, None, subtitle)
    else:
        plot_points(pdfmed, pdfmed-pdfmin, pdfmax-pdfmed, subtitle)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage = 'Usage: python plot_results.py <results filename>'
        print(usage)
        exit(1)
    filename = sys.argv[1]
    main(filename)
