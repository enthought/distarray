# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate the Julia set for a given z <- z**2 + c with NumPy. c is
passed via command line args. Usage:
    $ python julia_numpy.py <c real component> <c imaginary component>
"""


def calc_julia(fn, c, z_max, n_max, re_ax, im_ax):
    """Calculate of the function fn over the grid with the axes re_ax
    and im_ax

    Parameters
    ----------
    fn : function
        Function which accepts two complex positional arguments: z and
        c. And returns another complex number z. The standard quadratic
        julia set would use fn = lambda z, c: z*z + c.

    c : complex constant
       Typically the adjustable parameter when computing julia sets.

    z_max : real constant
        If abs(z) > z_max we deciede it diverges and stop iterating.

    n_max : int
        Maximum number of iterations for each point.

    re_ax : 1D numpy array
        NumPy array representing coordinates along the real axis.

    im_ax : 1D numpy array
        NumPy array representing coordinates along the imaginary
        axis.

    Returns
    -------
    A 2D numpy array with dimensions (re_ax.size, im_ax.size). With
    values representing the number of iterations at each point.
    """
    # output grid
    out = numpy.zeros((re_ax.size, im_ax.size))

    # Iterate over each pixel in our grid.
    for x, re in enumerate(re_ax):
        for y, im in enumerate(im_ax):
            z = complex(re, im)
            n = 0
            while abs(z) < z_max and n < n_max:
                z = fn(z, c)
                n += 1
            out[x, y] = n
    return out

if __name__ == '__main__':
    import sys

    from matplotlib import pyplot
    import numpy

    fn = lambda z, c: z*z + c
    c = complex(float(sys.argv[1]), float(sys.argv[2]))
    z_max = 10
    n_max = 100
    re_ax = numpy.linspace(-1.5, 1.5, 500)
    im_ax = numpy.linspace(-1.5, 1.5, 500)
    out = calc_julia(fn, c, z_max, n_max, re_ax, im_ax)

    pyplot.matshow(out.astype(numpy.float))
    pyplot.show()
