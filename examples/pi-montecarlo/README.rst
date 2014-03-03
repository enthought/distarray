Estimate pi using a Monte Carlo method
======================================

If we imagine a unit circle inscribed within a unit square, the ratio of the
area of the circle to the area of the square is pi/4. So if a point is chosen
at random within the square, it has a pi/4 probability of being inside the
circle too.

So we choose one N points in the square, count how many are in the circle and
divide by N to give an estimation of pi/4. We then multiply by 4 to get pi.

The convergence of this method is very slow O(n**-0.5).

- ``pi_numpy.py`` uses pure numpy to estimate pi.

- ``pi_ipython_parallel.py`` use IPython.parallel to estimate pi.

- ``pi_distarray.py`` uses distarray to estimate pi.

- ``benchmark.py`` benchmarks the methods.
