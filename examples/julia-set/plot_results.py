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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage = 'Usage: python plot_results.py <results filename>'
        print(usage)
        exit(1)
    filename = sys.argv[1]
    print('filename:', filename)
    # Read the file.
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        # Swallow header lines.
        a = csvreader.next()
        notes = csvreader.next()
        note_text = ','.join(notes)
        print('note_txt:', note_text)
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
            # Add some jitter to the engine count for less crowded plots.
            r = random.uniform(-0.125, +0.125)
            num_engines += r
            # Key for each curve.
            key = (dist, resolution)
            if key not in results:
                results[key] = {
                    'engines': [],
                    'times': [],
                    'legend': "%s %d" % (dist, resolution),
                }
            # Collect values to plot.
            results[key]['engines'].append(num_engines)
            results[key]['times'].append(t_distarray)
    # Sort keys for consistent coloring.
    keys = results.keys()
    keys.sort()
    # Get range of data for plot limits.
    eng_list = []
    tdist_list = []
    for key in keys:
        engines = results[key]['engines']
        times = results[key]['times']
        eng_list.extend(engines)
        tdist_list.extend(times)
    max_engines = max(eng_list)
    max_time = max(tdist_list)
    # Plot
    for key in keys:
        engines = results[key]['engines']
        times = results[key]['times']
        pyplot.plot(engines, times, 'o')
    pyplot.xlim((0, max_engines + 1))
    pyplot.ylim((0.0, 1.1 * max_time))
    pyplot.title('Julia Set Performance\n' + note_text)
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('DistArray time')
    legend = [results[key]['legend'] for key in keys]
    pyplot.legend(legend, loc='lower left')
    filename = 'julia_timing_plot.png'
    pyplot.savefig(filename, dpi=100)
    pyplot.show()
