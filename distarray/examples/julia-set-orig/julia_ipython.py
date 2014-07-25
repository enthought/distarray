# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
Calculate the Julia set for a given z <- z**2 + c with
IPython.parallel passed via command line args. Usage:
    $ python julia_ipython.py <c real component> <c imaginary component>
"""


def make_grid(re_ax, im_ax):
    """Create the grid we will build the fractal with."""
    out = numpy.empty((re_ax.size, im_ax.size), dtype=numpy.complex)
    for i, re in enumerate(re_ax):
        for j, im in enumerate(im_ax):
            out[i, j] = numpy.complex(re, im)
    return out


def push_grid(dview, grid):
    """Split the grid up into equal sections and scatter it on the
    engines.
    """
    for target, section in enumerate(numpy.hsplit(grid, len(dview))):
        dview.push(dict(grid=section), targets=target)


def pull_grid(dview, grid_name):
    """Collect the peices of our fractal."""
    out = numpy.hstack(dview[grid_name])
    return out


def push_setup(dview, fn, c, z_max, n_max, grid_name):
    """Push the Julia set parameters to the engines. And setup for
    Julia set calculations."""
    dview.push(dict(fn=fn, c=c, z_max=z_max, n_max=n_max))
    code = "{} = grid.copy()".format(grid_name)
    dview.execute(code)


def julia_loop(dview, grid_name):
    """Julia set calculations."""
    code = """
{} = grid.copy()
for (i,j), z in numpy.ndenumerate(grid):
    n = 0
    while abs(z) < z_max and n < n_max:
        z = fn(z, c)
        n += 1
    {}[i, j] = n
""".format(grid_name, grid_name)
    dview.execute(code)


def calc_julia(dview, fn, c, z_max, n_max, re_ax, im_ax):
    out_grid_name = 'out'
    # Make the grid.
    grid = make_grid(re_ax, im_ax)
    # scatter the grid across the engines.
    push_grid(dview, grid)
    # parameters for the calculations
    push_setup(dview, fn, c, z_max, n_max, out_grid_name)
    # Run the code on the engines.
    julia_loop(dview, out_grid_name)
    # Gather the peices
    out_grid = pull_grid(dview, out_grid_name)
    return out_grid

if __name__ == '__main__':
    import sys

    import numpy
    from IPython.parallel import Client
    from matplotlib import pyplot

    client = Client()
    dview = client[:]
    dview.block = True
    dview.execute("import numpy")

    fn = lambda z, c: z*z + c
    c = numpy.complex(float(sys.argv[1]), float(sys.argv[2]))
    z_max = 10
    n_max = 100
    re_ax = numpy.linspace(-1.5, 1.5, 500)
    im_ax = numpy.linspace(-1.5, 1.5, 500)
    out = calc_julia(dview, fn, c, z_max, n_max, re_ax, im_ax)

    pyplot.matshow(out.astype(numpy.float))
    pyplot.show()
