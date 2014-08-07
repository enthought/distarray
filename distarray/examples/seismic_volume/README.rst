Processing a Seismic Volume
===========================

This example shows some processing, using distarray, on a simulated
seismic volume. The volume can be stored either as an HDF5 file,
or as a set of .dnpy files.

If you have an MPI-enabled HDF5 library, you can use that for loading
and saving these files. Otherwise you can use our .dnpy flat file format
with no external dependencies. By default, the HDF5 file format is used,
but you can pass ``--dnpy`` as a command line argument to both scripts
to use the flat file format instead.

- ``create_volume.py`` - This script creates the simulated seismic
volume. The internal structure of the volume includes two 'horizons',
which are plane surfaces inside the volume. As one moves down in depth,
approaching and then crossing the horizon, the data values increase
with a Gaussian peak at the horizon, and then settle down to a larger
value than above the horizon.

If you would like array dimensions other than the default, you can pass
``--size NX NY NZ`` as command line arguments to get a volume with shape
(NX, NY, NZ). Other values in the script can also be modified to change
the values in the volume array.

The seismic volume is written either to an HDF5 named seismic.hdf5, or
a set of .dnpy files named seismic_0.dnpy, seismic_1.dnpy, and so on.

- ``load_volume.py`` - This script loads the volume, and performs several
processing steps on it. First, the seismic volume is loaded in parallel
into a distarray. The script then loops over all of the traces (z-slices
constant in x and y), and calculates some statistics on each trace.
(I.e. min, max, average, standard deviation). Next, the script applies
a couple of filters on each trace. The first filter is a 3-point average,
and the second filter is a 3-point window maximum. Then, the script extracts
three slices from the seismic volume, on all three principal axes, and
creates a plot for each slice. The script also extracts slices and makes
plots on the result of the 3-point average filter. Finally, the result
of the 3-point average filter is written out, both to an HDF5 file,
and to a set of .dnpy files.

The script will also perform many of these operations using NumPy directly,
and check that it gets the same results as when operating on the distributed
array.
