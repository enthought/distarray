# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

'''
Plot the results of the Julia set timings.
'''

from __future__ import print_function

import sys
import csv
import random
from matplotlib import pyplot


# Dictionary keys.
ENGINES = 'engines'
TIMES = 'times'
LEGEND = 'legend'


def read_results(filename):
    ''' Read the Julia Set timing results from the file. '''
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        # Swallow header lines.
        a = csvreader.next()
        notes = csvreader.next()
        note_text = ','.join(notes)
        # And the field names.
        c = csvreader.next()
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
                    LEGEND: "%s %d" % (dist, resolution),
                }
            # Collect values to plot.
            results[key][ENGINES].append(num_engines)
            results[key][TIMES].append(t_distarray)
    return results, note_text


def jitter_engines(results, amount):
    ''' Apply some random jitter to the integer engine count,
    to make less crowded looking plot. '''
    for key in results:
        engines = results[key][ENGINES]
        engines = [engine + random.uniform(-amount, +amount) for engine in engines]
        results[key][ENGINES] = engines


def trim_results(results):
    ''' Select only the smallest time, consistent with timeit. '''
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
    ''' Get the range of the data (for plot limits). '''
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


def plot_results(filename, results, subtitle, x_min, x_max, y_min, y_max):
    ''' Plot the timing results. '''
    # Sort keys for consistent coloring.
    keys = results.keys()
    keys.sort()
    for key in keys:
        engines = results[key][ENGINES]
        times = results[key][TIMES]
        pyplot.plot(engines, times, 'o-')
    pyplot.xlim((x_min, x_max))
    pyplot.ylim((y_min, y_max))
    pyplot.title('Julia Set Performance\n' + subtitle)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('DistArray time')
    legend = [results[key][LEGEND] for key in keys]
    pyplot.legend(legend, loc='lower left')
    pyplot.savefig(filename, dpi=100)
    pyplot.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage = 'Usage: python plot_results.py <results filename>'
        print(usage)
        exit(1)
    filename = sys.argv[1]
    # Read and parse timing results.
    results, note_text = read_results(filename)
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
    plot_results(filename, results, subtitle, x_min, x_max, y_min, y_max)
