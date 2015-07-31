README: Julia Set
===============

This example calculates some
[Julia sets](https://en.wikipedia.org/wiki/Julia_set) using distarray and
measures the performance. The code provides the user a choice of the kernel to
use for computation. Unsurprisingly, the fastest kernel turns out to be the
Cython kernel which must be compiled before we can perform any computation:

`python setup.py build_ext --inplace`

With the Cython kernel ready to go, we use the `benchmark_julia.py` script to
generate our performance data, the usage of which is as follows:

```
usage: benchmark_julia.py [-h] [-r REPEAT_COUNT] [-o OUTPUT_FILENAME]
[-k {fancy,numpy,cython}] [-s {strong,weak}] N [N ...]

positional arguments:
N                     resolutions of the Julia set to benchmark (NxN)

optional arguments:
-h, --help          show this help message and exit
-r REPEAT_COUNT, --repeat REPEAT_COUNT
                    number of repetitions of each unique parameter set,
                    default: 3
-o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                    filename to write the json data to.
-k {fancy,numpy,cython}, --kernel {fancy,numpy,cython}
                    kernel to use for computation. Options are 'fancy',
                    'numpy', or 'cython'.
-s {strong,weak}, --scaling {strong,weak}
                    Kind of scaling test. Options are 'strong' or 'weak'
```

This script can be invoked as follows:

```
mpiexec -np NPROC python benchmark_julia.py  [-h] [-r REPEAT_COUNT]
        [-o OUTPUT_FILENAME] [-k {fancy,numpy,cython}]
        [-s {strong,weak}] N [N ...]
```

Where `NPROC` is the number of MPI processes to be used
(`1` client + `N-1` workers). The results contained in the `OUTPUT_FILENAME`
can be plotted using the `plot_results.py` script:

```
python plot_results.py OUTPUT_FILENAME
```

The `mpi_benchmark ` script can also be used to drive program execution.
Here the number of MPI processes is hardcoded into the script.
Command-line usage of the script is as follows:

```
usage: ./mpi_benchmark <reps> <output_filename> <kernel> <resolution>
```
