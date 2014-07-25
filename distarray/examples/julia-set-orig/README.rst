Calculating the Julia Set
=========================

This demo calculates the Julia Set for a given function using several
different python packages, including distarray.

- ``julia_numpy.py`` the most naive serial approach using NumPy.

- ``julia_ipython.py`` Parallelize computations using IPython.parallel
  which uses block distributed arrays.

- ``julia_distarray.py`` Parallelize the computations using distarray
  with a block-cyclic array layout.

- ``bench_dist.py`` Benchmark array distributions and engines numbers
  for calculating the Julia set with Distarray.
