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
import csv
import random
import numpy

import matplotlib
from matplotlib import pyplot

from distarray.externals.six import next


CBcdict={
    'Bl':(0,0,0),
    'Or':(.9,.6,0),
    'SB':(.35,.7,.9),
    'bG':(0,.6,.5),
    'Ye':(.95,.9,.25),
    'Bu':(0,.45,.7),
    'Ve':(.8,.4,0),
    'rP':(.8,.6,.7),
}

#Change default color cycle
matplotlib.rcParams['axes.color_cycle'] = [CBcdict[c] for c in sorted(CBcdict.keys())]


# Dictionary keys.
ENGINES = 'engines'
TIMES = 'times'
LEGEND = 'legend'
RESOLUTION = 'resolution'


STYLES = ('o-', 'x-', 'v-', '*-', 's-', 'd-')

def read_results(filename):
    """Read the Julia Set timing results from the file."""
    with open(filename, 'rt') as csvfile:
        csvreader = csv.reader(csvfile)
        # Swallow header lines.
        # Discard 'importing numpy on engines'
        next(csvreader)
        # Title
        title_fields = next(csvreader)
        title = title_fields[0]
        # Subtitle/notes
        notes = next(csvreader)
        note_text = ','.join(notes)
        # Field names.
        next(csvreader)
        # Read the results.
        results = {}
        for row in csvreader:
            dist = row[0]
            num_engines = int(row[1])
            resolution = int(row[2])
            t_distarray = float(row[3])
            t_numpy = float(row[4])
            t_ratio = float(row[5])
            iters = float(row[6])
            c = row[7]    # As a string.
            # Key for each curve.
            key = (dist, resolution)
            if key not in results:
                results[key] = {
                    ENGINES: [],
                    TIMES: [],
                    RESOLUTION: resolution,
                    LEGEND: "%s %d" % (dist, resolution),
                }
            # Collect values to plot.
            results[key][ENGINES].append(num_engines)
            results[key][TIMES].append(t_distarray)
    return results, title, note_text


def jitter_engines(results, amount):
    """Apply some random jitter to the integer engine count,
    to make less crowded looking plot.
    """
    for key in results:
        engines = results[key][ENGINES]
        engines = [engine + random.uniform(-amount, +amount)
                   for engine in engines]
        results[key][ENGINES] = engines


def trim_results(results):
    """Select only the smallest time, consistent with timeit."""
    for key in results:
        engines = results[key][ENGINES]
        times = results[key][TIMES]
        trim = {}
        for engine, time in zip(engines, times):
            if engine not in trim:
                trim[engine] = []
            trim[engine].append(time)
        for engine in trim:
            times = trim[engine]
            min_time = min(times)
            trim[engine] = min_time
        trimmed_engines = []
        trimmed_times = []
        for engine in trim:
            trimmed_engines.append(engine)
            trimmed_times.append(trim[engine])
        # Sort by engine count for better line plot,
        # and sort times to match.
        sorted_engines, sorted_times = zip(*sorted(zip(trimmed_engines,
                                                       trimmed_times)))
        results[key][ENGINES] = sorted_engines
        results[key][TIMES] = sorted_times


def get_results_range(results):
    """Get the range of the data (for plot limits)."""
    all_engines = []
    all_times = []
    for key in results:
        engines = results[key][ENGINES]
        times = results[key][TIMES]
        all_engines.extend(engines)
        all_times.extend(times)
    max_engine = max(all_engines)
    max_time = max(all_times)
    return max_engine, max_time


def plot_results(filename, results, title, subtitle, x_min, x_max, y_min, y_max):
    """Plot the timing results."""
    # Sort keys for consistent coloring.
    keys = results.keys()
    keys = sorted(keys)
    styles = iter(STYLES)
    for key in keys:
        engines = results[key][ENGINES]
        times = results[key][TIMES]
        pyplot.plot(engines, times, next(styles))
    pyplot.xlim((x_min, x_max))
    pyplot.ylim((y_min, y_max))
    full_title = title + '\n' + subtitle
    pyplot.title(full_title)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('DistArray time')
    legend = [results[key][LEGEND] for key in keys]
    pyplot.legend(legend, loc='lower left')
    pyplot.savefig(filename, dpi=100)
    pyplot.show()


def plot_points(filename, results, title, subtitle, ideal_dist=('b-b', 512)):
    """Plot the timing results."""
    # Sort keys for consistent coloring.
    keys = results.keys()
    keys = sorted(keys)
    styles = iter(STYLES)
    for key in keys:
        engines = results[key][ENGINES]
        times = results[key][TIMES]
        npoints = (results[key][RESOLUTION] ** 2) / numpy.array(times)
        pyplot.plot(engines, npoints / 1000, next(styles), markersize=10)

    # plot idealized scaling
    ideal_line_base = (results[ideal_dist][RESOLUTION]**2) / results[ideal_dist][TIMES][0]
    ideal_line = ideal_line_base * numpy.array(results[ideal_dist][ENGINES])
    pyplot.plot(engines, ideal_line / 1000, '--')

    pyplot.xlim((min(engines)-1, max(engines)+1))
    full_title = title + '\n' + subtitle
    pyplot.title(full_title)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('kpoints / s')
    legend = [results[key][LEGEND] for key in keys]
    pyplot.legend(legend, loc='lower right')
    pyplot.savefig(filename, dpi=100)
    pyplot.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage = 'Usage: python plot_results.py <results filename>'
        print(usage)
        exit(1)
    filename = sys.argv[1]
    # Read and parse timing results.
    results, title, note_text = read_results(filename)
    # Either pick just the minimum time, or add jitter to the engine count.
    if True:
        trim_results(results)
    else:
        jitter_engines(results, 0.125)
    # Get range of data for plot limits.
    max_engines, max_time = get_results_range(results)
    # Plot
    filename = 'julia_timing_plot.png'
    subtitle = note_text
    x_min, x_max = 0, max_engines + 1
    y_min, y_max = 0.0, 1.1 * max_time
    #plot_results(filename, results, title, subtitle, x_min, x_max, y_min, y_max)
    plot_points(filename, results, title, subtitle)
