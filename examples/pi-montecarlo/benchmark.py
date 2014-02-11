"""
Benchmark all the scripts. Usage:
    $ python benchmark.py <n>

Where in is the power of 2 for the maximum number of points
benchmarked. i.e. N = 2**n where N is the number of points used.
"""
import sys
import timeit

from matplotlib import pyplot as plt

import pi_numpy
import pi_distarray
import pi_ipython_parallel


def bench(module, n_list):
    times = []
    module_name = module.__name__
    for n in n_list:
        t = timeit.timeit("{}.calc_pi({})".format(module_name, n),
                          setup="from __main__ import {}".format(module_name),
                          number=1)
        times.append(t)
    return times

if __name__ == '__main__':
    # n is maximum order of magnitude for the number of iterations.
    n = int(sys.argv[1])
    n_list = [2**i for i in range(3, n)]

    # Do the benchmarking.
    print("Benchmarking IPython.parallel")
    ipython_parallel_times = bench(pi_ipython_parallel, n_list)
    pi_ipython_parallel.client.close()
    print("Benchmarking NumPy")
    numpy_times = bench(pi_numpy, n_list)
    print("Benchmarking distarray")
    distarray_times = bench(pi_distarray, n_list)
    pi_distarray.context.client.close()

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_list, distarray_times, lw=2, label='distarray')
    ax.plot(n_list, ipython_parallel_times, lw=2, label='IPython.parallel')
    ax.plot(n_list, numpy_times, lw=2, label='NumPy')

    # annotations
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of iterations (n)')
    ax.set_ylabel('seconds (s)')
    ax.legend(loc='upper left')
    ax.set_title('Benchmark Monte Carlo Estimation of $\pi$')

    # Save to plot.png.
    plt.savefig('plot')
