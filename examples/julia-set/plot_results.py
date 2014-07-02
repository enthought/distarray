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
        csvreader.next()
        csvreader.next()
        # And the field names.
        csvreader.next()
        # Read the results.
        eng_b_list = []
        tdist_b_list = []
        eng_c_list = []
        tdist_c_list = []
        for row in csvreader:
            dist = row[0]
            num_engines = int(row[1])
            resolution = int(row[2])
            t_distarray = float(row[3])
            t_numpy = float(row[4])
            t_ratio = float(row[5])
            iters = float(row[6])
            c = row[7]    # As a string.
            #print(dist, num_engines, resolution, t_distarray, t_numpy, t_ratio, iters, c)
            # Add some jitter to the engine count for less crowded plots.
            r = random.uniform(-0.125, +0.125)
            num_engines += r
            # Collect values to plot.
            if dist == 'b':
                eng_b_list.append(num_engines)
                tdist_b_list.append(t_distarray)
            elif dist == 'c':
                eng_c_list.append(num_engines)
                tdist_c_list.append(t_distarray)
    # Get range of data for plot limits.
    eng_list = []
    eng_list.extend(eng_b_list)
    eng_list.extend(eng_c_list)
    tdist_list = []
    tdist_list.extend(tdist_b_list)
    tdist_list.extend(tdist_c_list)
    # Extents.
    max_engines = max(eng_list)
    max_time = max(tdist_list)
    # Plot
    pyplot.plot(eng_b_list, tdist_b_list, 'bo')
    pyplot.plot(eng_c_list, tdist_c_list, 'ro')
    pyplot.xlim((0, max_engines + 1))
    pyplot.ylim((0.0, 1.1 * max_time))
    pyplot.title('Julia Set Performance')
    pyplot.xlabel('Engine Count')
    pyplot.ylabel('DistArray time')
    pyplot.legend(("'b' distribution", "'c' distribution"), loc='lower left')
    filename = 'julia_timing_plot.png'
    pyplot.savefig(filename, dpi=100)
    pyplot.show()
