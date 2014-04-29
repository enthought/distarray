# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------
"""
`Context` objects contain the information required for distarrays to
communicate with localarrays.
"""


import uuid
import collections
import atexit

import numpy

from distarray import cleanup
from distarray.externals import six
from distarray.client import DistArray
from distarray.client_map import Distribution
from distarray.ipython_utils import IPythonClient


DISTARRAY_BASE_NAME = '__distarray__'


class Context(object):
    """
    Context objects manage the setup and communication of the worker processes
    for DistArray objects.  A DistArray object has a context, and contexts have
    an MPI intracommunicator that they use to communicate with worker
    processes.

    Typically there is just one context object that uses all processes,
    although it is possible to have more than one context with a different
    selection of engines.

    """

    _CLEANUP = None

    def __init__(self, client=None, targets=None):

        if not Context._CLEANUP:
            Context._CLEANUP = (atexit.register(cleanup.clear_all),
                                atexit.register(cleanup.cleanup_all, '__main__', DISTARRAY_BASE_NAME))

        if client is None:
            self.client = IPythonClient()
            self.owns_client = True
        else:
            self.client = client
            self.owns_client = False

        self.view = self.client[:]

        all_targets = self.view.targets
        if targets is None:
            self.targets = all_targets
        else:
            self.targets = []
            for target in targets:
                if target not in all_targets:
                    raise ValueError("Engine with id %r not registered" % target)
                else:
                    self.targets.append(target)

        # FIXME: IPython bug #4296: This doesn't work under Python 3
        #with self.view.sync_imports():
        #    import distarray
        self.view.execute("import distarray.local; "
                          "import distarray.mpiutils; "
                          "import numpy")

        self.context_key = self._setup_context_key()
        self._comm_key = self._make_intracomm()
        self._set_engine_rank_mapping()

    def _setup_context_key(self):
        """
        Create a dict on the engines which will hold everything from
        this context.
        """
        context_key = DISTARRAY_BASE_NAME + self.uid()
        cmd = ("import types, sys;"
               "%s = types.ModuleType('%s');")
        cmd %= (context_key, context_key)
        self._execute(cmd, targets=range(len(self.view)))
        return context_key

    def _make_intracomm(self):
        def get_rank():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_rank()

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_size()

        # get a mapping of IPython engine ID to MPI rank
        rank_map = self.view.apply_async(get_rank).get_dict()
        ranks = [ rank_map[engine] for engine in self.targets ]

        comm_size = self.view.apply_async(get_size).get()[0]
        if set(rank_map.values()) != set(range(comm_size)):
            raise ValueError('Engines in view must encompass all MPI ranks.')

        # create a new communicator with the subset of engines note that
        # MPI_Comm_create must be called on all engines, not just those
        # involved in the new communicator.
        comm_key = self._generate_key()
        cmd = "%s = distarray.mpiutils.create_comm_with_list(%s)"
        cmd %= (comm_key, ranks)
        self.view.execute(cmd, block=True)
        return comm_key

    def _set_engine_rank_mapping(self):
        # The MPI intracomm referred to by self._comm_key may have a different
        # mapping between IPython engines and MPI ranks than COMM_PRIVATE.  We
        # reorder self.targets so self.targets[i] is the IPython engine ID that
        # corresponds to MPI intracomm rank i.
        rank = self._generate_key()
        self.view.execute(
                '%s = %s.Get_rank()' % (rank, self._comm_key),
                block=True, targets=self.targets)

        # mapping target -> rank, rank -> target.
        rank_from_target = self.view.pull(rank, targets=self.targets).get_dict()
        target_from_rank = {v: k for (k, v) in rank_from_target.items()}

        # ensure consistency
        assert set(self.targets) == set(rank_from_target)
        assert set(range(len(self.targets))) == set(target_from_rank)

        # reorder self.targets so that the targets are in MPI rank order for
        # the intracomm.
        self.targets = [target_from_rank[i] for i in range(len(target_from_rank))]

    # Key management routines:
    @staticmethod
    def _key_prefix():
        """ Get the base name for all keys. """
        return DISTARRAY_BASE_NAME

    @staticmethod
    def uid():
        """Generate a unique valid python name."""
        # Full length seems excessively verbose so use 16 characters.
        return Context._key_prefix() + uuid.uuid4().hex[:16]

    def _generate_key(self):
        """ Generate a unique key name for this context. """
        key = "%s.%s" % (self.context_key, 'key_' + self.uid())
        return key

    def _key_and_push(self, *values):
        keys = [self._generate_key() for value in values]
        self._push(dict(zip(keys, values)))
        return tuple(keys)

    def delete_key(self, key):
        """ Delete the specific key from all the engines. """
        cmd = 'del %s' % key
        self._execute(cmd)

    def cleanup(self):
        """ Delete keys that this context created from all the engines. """
        cleanup.cleanup(view=self.view, module_name='__main__', prefix=self._key_prefix())

    def close(self):
        self.cleanup()
        if self.owns_client:
            self.client.close()
        self._comm_key = None

    # End of key management routines.

    def _execute(self, lines, targets=None):
        targets = targets or self.targets
        return self.view.execute(lines, targets=targets, block=True)

    def _push(self, d, targets=None):
        targets = targets or self.targets
        return self.view.push(d, targets=targets, block=True)

    def _pull(self, k, targets=None):
        targets = targets or self.targets
        return self.view.pull(k, targets=targets, block=True)

    def _execute0(self, lines):
        return self.view.execute(lines, targets=self.targets[0], block=True)

    def _push0(self, d):
        return self.view.push(d, targets=self.targets[0], block=True)

    def _pull0(self, k):
        return self.view.pull(k, targets=self.targets[0], block=True)

    def _create_local(self, local_call, shape, dist, grid_shape, dtype):
        """ Creates a local array, according to the method named in `local_call`."""
        shape_name, dist_name, grid_shape_name, dtype_name = \
                self._key_and_push(shape, dist, grid_shape, dtype)
        da_key = self._generate_key()
        comm_key = self._comm_key
        cmd = '{da_key} = {local_call}(distarray.local.maps.Distribution.from_shape({shape_name}, {dist_name}, {grid_shape_name}, {comm_key}), {dtype_name})'
        self._execute(cmd.format(**locals()))
        return DistArray.from_localarrays(da_key, self)

    def zeros(self, shape, dtype=float, dist=None, grid_shape=None):
        if dist is None:
            dist = {0: 'b'}
        return self._create_local(local_call='distarray.local.zeros',
                                  shape=shape, dist=dist,
                                  grid_shape=grid_shape, dtype=dtype)

    def ones(self, shape, dtype=float, dist=None, grid_shape=None):
        if dist is None:
            dist = {0: 'b'}
        return self._create_local(local_call='distarray.local.ones',
                                  shape=shape, dist=dist,
                                  grid_shape=grid_shape, dtype=dtype,)

    def empty(self, shape, dtype=float, dist=None, grid_shape=None):
        if dist is None:
            dist = {0: 'b'}
        return self._create_local(local_call='distarray.local.empty',
                                  shape=shape, dist=dist,
                                  grid_shape=grid_shape, dtype=dtype)

    def from_global_dim_data(self, global_dim_data, dtype=float):
        """Make a DistArray from global dim_data structures.

        Parameters
        ----------
        global_dim_data : tuple of dict
            A global dimension dictionary per dimension.  See following `Note`
            section.
        dtype : numpy dtype, optional
            dtype for underlying arrays

        Returns
        -------
        result : DistArray
            An empty DistArray of the specified size, dimensionality, and
            distribution.

        Note
        ----

        The `global_dim_data` tuple is a simple, straightforward data structure
        that allows full control over all aspects of a DistArray's distribution
        information.  It does not contain any of the array's *data*, only the
        *metadata* needed to specify how the array is to be distributed.  Each
        dimension of the array is represented by corresponding dictionary in
        the tuple, one per dimension.  All dictionaries have a `dist_type` key
        that specifies whether the array is block, cyclic, or unstructured.
        The other keys in the dictionary are dependent on the `dist_type` key.

        **Block**

        * ``dist_type`` is ``'b'``.

        * ``bounds`` is a sequence of integers, at least two elements.

          The ``bounds`` sequence always starts with 0 and ends with the global
          ``size`` of the array.  The other elements indicate the local array
          global index boundaries, such that successive pairs of elements from
          ``bounds`` indicates the ``start`` and ``stop`` indices of the
          corresponding local array.

        * ``comm_padding`` integer, greater than or equal to zero.
        * ``boundary_padding`` integer, greater than or equal to zero.

        These integer values indicate the communication or boundary padding,
        respectively, for the local arrays.  Currently only a single value for
        both ``boundary_padding`` and ``comm_padding`` is allowed for the
        entire dimension.

        **Cyclic**

        * ``dist_type`` is ``'c'``

        * ``proc_grid_size`` integer, greater than or equal to one.

        The size of the process grid in this dimension.  Equivalent to the
        number of local arrays in this dimension and determines the number of
        array sections.

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.

        * ``block_size`` integer, optional.  Greater than or equal to one.

        If not present, equivalent to being present with value of one.

        **Unstructured**

        * ``dist_type`` is ``'u'``

        * ``indices`` sequence of one-dimensional numpy integer arrays or
          buffers.

          The ``len(indices)`` is the number of local unstructured arrays in
          this dimension.

          To compute the global size of the array in this dimension, compute
          ``sum(len(ii) for ii in indices)``.

        **Not-distributed**

        The ``'n'`` distribution type is a convenience to specify that an array
        is not distributed along this dimension.

        * ``dist_type`` is ``'n'``

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.

        """
        # global_dim_data is a sequence of dictionaries, one per dimension.
        distribution = Distribution(self, global_dim_data)
        dim_data_per_rank = distribution.get_dim_data_per_rank()

        if len(self.targets) != len(dim_data_per_rank):
            errmsg = "`dim_data_per_rank` must contain a dim_data for every rank."
            raise TypeError(errmsg)

        da_key = self._generate_key()
        subs = ((da_key,) + self._key_and_push(dim_data_per_rank) +
                (self._comm_key, self._comm_key) + self._key_and_push(dtype))

        cmd = ('%s = distarray.local.LocalArray('
                    'distarray.local.maps.Distribution(%s[%s.Get_rank()], comm=%s),'
                    ' dtype=%s)')
        self._execute(cmd % subs)

        return DistArray.from_localarrays(da_key, self)

    def save_dnpy(self, name, da):
        """
        Save a distributed array to files in the ``.dnpy`` format.

        The ``.dnpy`` file format is a binary format inspired by NumPy's
        ``.npy`` format.  The header of a particular ``.dnpy`` file contains
        information about which portion of a DistArray is saved in it (using
        the metadata outlined in the Distributed Array Protocol), and the data
        portion contains the output of NumPy's `save` function for the local
        array data.  See the module docstring for `distarray.local.format` for
        full details.

        Parameters
        ----------
        name : str or list of str
            If a str, this is used as the prefix for the filename used by each
            engine.  Each engine will save a file named ``<name>_<rank>.dnpy``.
            If a list of str, each engine will use the name at the index
            corresponding to its rank.  An exception is raised if the length of
            this list is not the same as the context's communicator's size.
        da : DistArray
            Array to save to files.

        Raises
        ------
        TypeError
            If `name` is an sequence whose length is different from the
            context's communicator's size.

        See Also
        --------
        load_dnpy : Loading files saved with save_dnpy.

        """
        if isinstance(name, six.string_types):
            subs = self._key_and_push(name) + (da.key, da.key)
            self._execute(
                'distarray.local.save_dnpy(%s + "_" + str(%s.comm_rank) + ".dnpy", %s)' % subs
            )
        elif isinstance(name, collections.Sequence):
            if len(name) != len(self.targets):
                errmsg = "`name` must be the same length as `self.targets`."
                raise TypeError(errmsg)
            subs = self._key_and_push(name) + (da.key, da.key)
            self._execute(
                'distarray.local.save_dnpy(%s[%s.comm_rank], %s)' % subs
            )
        else:
            errmsg = "`name` must be a string or a list."
            raise TypeError(errmsg)


    def load_dnpy(self, name):
        """
        Load a distributed array from ``.dnpy`` files.

        The ``.dnpy`` file format is a binary format inspired by NumPy's
        ``.npy`` format.  The header of a particular ``.dnpy`` file contains
        information about which portion of a DistArray is saved in it (using
        the metadata outlined in the Distributed Array Protocol), and the data
        portion contains the output of NumPy's `save` function for the local
        array data.  See the module docstring for `distarray.local.format` for
        full details.

        Parameters
        ----------
        name : str or list of str
            If a str, this is used as the prefix for the filename used by each
            engine.  Each engine will load a file named ``<name>_<rank>.dnpy``.
            If a list of str, each engine will use the name at the index
            corresponding to its rank.  An exception is raised if the length of
            this list is not the same as the context's communicator's size.

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded on each engine.

        Raises
        ------
        TypeError
            If `name` is an iterable whose length is different from the
            context's communicator's size.

        See Also
        --------
        save_dnpy : Saving files to load with with load_dnpy.

        """
        da_key = self._generate_key()

        if isinstance(name, six.string_types):
            subs = (da_key,) + self._key_and_push(name) + (self._comm_key,
                    self._comm_key)
            self._execute(
                '%s = distarray.local.load_dnpy(%s + "_" + str(%s.Get_rank()) + ".dnpy", %s)' % subs
            )
        elif isinstance(name, collections.Sequence):
            if len(name) != len(self.targets):
                errmsg = "`name` must be the same length as `self.targets`."
                raise TypeError(errmsg)
            subs = (da_key,) + self._key_and_push(name) + (self._comm_key,
                    self._comm_key)
            self._execute(
                '%s = distarray.local.load_dnpy(%s[%s.Get_rank()], %s)' % subs
            )
        else:
            errmsg = "`name` must be a string or a list."
            raise TypeError(errmsg)

        return DistArray.from_localarrays(da_key, self)

    def save_hdf5(self, filename, da, key='buffer', mode='a'):
        """
        Save a DistArray to a dataset in an ``.hdf5`` file.

        Parameters
        ----------
        filename : str
            Name of file to write to.
        da : DistArray
            Array to save to a file.
        key : str, optional
            The identifier for the group to save the DistArray to (the default
            is 'buffer').
        mode : optional, {'w', 'w-', 'a'}, default 'a'

            ``'w'``
                Create file, truncate if exists
            ``'w-'``
                Create file, fail if exists
            ``'a'``
                Read/write if exists, create otherwise (default)

        """
        try:
            # this is just an early check,
            # h5py isn't necessary until the local call on the engines
            import h5py
        except ImportError:
            errmsg = "An MPI-enabled h5py must be available to use save_hdf5."
            raise ImportError(errmsg)

        subs = (self._key_and_push(filename) + (da.key,) +
                self._key_and_push(key, mode))
        self._execute(
            'distarray.local.save_hdf5(%s, %s, %s, %s)' % subs
        )

    def load_npy(self, filename, dim_data_per_rank):
        """
        Load a DistArray from a dataset in a ``.npy`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        dim_data_per_rank : sequence of tuples of dict
            A "dim_data" data structure for every rank.  Described here:
            https://github.com/enthought/distributed-array-protocol

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded.

        """
        if len(self.targets) != len(dim_data_per_rank):
            errmsg = "`dim_data_per_rank` must contain a dim_data for every rank."
            raise TypeError(errmsg)

        da_key = self._generate_key()
        subs = ((da_key,) + self._key_and_push(filename, dim_data_per_rank) +
                (self._comm_key,) + (self._comm_key,))

        self._execute(
            '%s = distarray.local.load_npy(%s, %s[%s.Get_rank()], %s)' % subs
        )

        return DistArray.from_localarrays(da_key, self)

    def load_hdf5(self, filename, dim_data_per_rank, key='buffer'):
        """
        Load a DistArray from a dataset in an ``.hdf5`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        dim_data_per_rank : sequence of tuples of dict
            A "dim_data" data structure for every rank.  Described here:
            https://github.com/enthought/distributed-array-protocol
        key : str, optional
            The identifier for the group to load the DistArray from (the
            default is 'buffer').

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded.

        """
        try:
            import h5py
        except ImportError:
            errmsg = "An MPI-enabled h5py must be available to use load_hdf5."
            raise ImportError(errmsg)

        if len(self.targets) != len(dim_data_per_rank):
            errmsg = "`dim_data_per_rank` must contain a dim_data for every rank."
            raise TypeError(errmsg)

        da_key = self._generate_key()
        subs = ((da_key,) + self._key_and_push(filename, dim_data_per_rank) +
                (self._comm_key,) + self._key_and_push(key) + (self._comm_key,))

        self._execute(
            '%s = distarray.local.load_hdf5(%s, %s[%s.Get_rank()], %s, %s)' % subs
        )

        return DistArray.from_localarrays(da_key, self)

    def fromndarray(self, arr, dist=None, grid_shape=None):
        """Convert an ndarray to a distarray."""
        if dist is None:
            dist = {0: 'b'}
        out = self.empty(arr.shape, dtype=arr.dtype, dist=dist,
                         grid_shape=grid_shape)
        for index, value in numpy.ndenumerate(arr):
            out[index] = value
        return out

    fromarray = fromndarray

    def fromfunction(self, function, shape, **kwargs):
        func_key = self._generate_key()
        self.view.push_function({func_key: function}, targets=self.targets,
                                block=True)
        keys = self._key_and_push(shape, kwargs)
        new_key = self._generate_key()
        subs = (new_key, func_key) + keys
        self._execute('%s = distarray.local.fromfunction(%s,%s,**%s)' % subs)
        return DistArray.from_localarrays(new_key, self)
