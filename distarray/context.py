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
from distarray.externals import six
import collections

import numpy

from distarray.client import DistArray
from distarray.client_map import ClientMDMap
from distarray.ipython_utils import IPythonClient


class Context(object):
    '''
    Context objects manage the setup and communication of the worker processes
    for DistArray objects.  A DistArray object has a context, and contexts have
    an MPI intracommunicator that they use to communicate with worker
    processes.

    Typically there is just one context object that uses all processes,
    although it is possible to have more than one context with a different
    selection of engines.

    '''

    def __init__(self, client=None, targets=None):
        self.client = client if client is not None else IPythonClient()
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

        self._setup_key_context()
        self._make_intracomm()
        self._set_engine_rank_mapping()

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

    def _make_intracomm(self):
        def get_rank():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_rank()

        # get a mapping of IPython engine ID to MPI rank
        rank_map = self.view.apply_async(get_rank).get_dict()
        ranks = [ rank_map[engine] for engine in self.targets ]

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_size()

        comm_size = self.view.apply_async(get_size).get()[0]
        if set(rank_map.values()) != set(range(comm_size)):
            raise ValueError('Engines in view must encompass all MPI ranks.')

        # create a new communicator with the subset of engines note that
        # MPI_Comm_create must be called on all engines, not just those
        # involved in the new communicator.
        self._comm_key = self._generate_key()
        self.view.execute(
            '%s = distarray.mpiutils.create_comm_with_list(%s)' % (self._comm_key, ranks),
            block=True
        )

    # Key management routines:

    def _setup_key_context(self):
        """ Generate a unique string for this context.

        This will be included in the names of all keys we create.
        This prefix allows us to delete only keys from this context.
        """
        # Full length seems excessively verbose so use 16 characters.
        uid = uuid.uuid4()
        self.key_context = uid.hex[:16]

    def _key_basename(self):
        """ Get the base name for all keys. """
        return '_distarray_key'

    def _key_prefix(self):
        """ Generate a prefix for a key name for this context. """
        header = self._key_basename() + '_' + self.key_context
        return header

    def _generate_key(self):
        """ Generate a unique key name for this context. """
        uid = uuid.uuid4()
        key = self._key_prefix() + '_' + uid.hex
        return key

    def _key_and_push(self, *values):
        keys = [self._generate_key() for value in values]
        self._push(dict(zip(keys, values)))
        return tuple(keys)

    def delete_key(self, key):
        """ Delete the specific key from all the engines. """
        cmd = 'del %s' % key
        self._execute(cmd)

    def cleanup_keys(self, all_other_contexts=False):
        """ Delete keys that this context created from all the engines.

        If all_other_contexts is False (the default), then this
        deletes from the engines all the keys from only this context.
        Otherwise, it deletes all keys from all other contexts.

        """
        # make the _comm_key invalid
        self._comm_key = None
        basename = self._key_basename()
        prefix = self._key_prefix()
        if all_other_contexts:
            # Delete distarray keys from all contexts except this one.
            cmd = """for k in list(globals().keys()):
                         if (k.startswith('%s')) and (not k.startswith('%s')):
                             del globals()[k]""" % (basename, prefix)
        else:
            # Delete keys only from this context.
            cmd = """for k in list(globals().keys()):
                         if k.startswith('%s'):
                             del globals()[k]""" % (prefix)
        self._execute(cmd)

    def dump_keys(self, all_other_contexts=False):
        """ Return a list of the key names present on the engines.

        If all_other_contexts is False (the default), then this
        returns only the keys for this context.
        Otherwise, it returns the keys for all other contexts.

        The list is a list of tuples (key name, list of targets),
        and is sorted by key name. This is intended to be convenient
        and readable to print out.
        """
        dump_key = self._generate_key()
        cmd = '%s = [k for k in globals().keys() if k.startswith("%s")]' % (
            dump_key, self._key_basename())
        self._execute(cmd)
        keylists = self._pull(dump_key)
        # The values returned by the engines are a nested list,
        # the outer per engine, and the inner listing each key name.
        # Convert to dict with key=key, value=list of targets.
        engine_keys = {}
        prefix = self._key_prefix()
        for iengine, keylist in enumerate(keylists):
            for key in keylist:
                # Limit to the keys we care about.
                if not all_other_contexts:
                    # Skip keys not from this context.
                    if not key.startswith(prefix):
                        continue
                else:
                    # Skip keys from this context.
                    if key.startswith(prefix):
                        continue
                if key not in engine_keys:
                    engine_keys[key] = []
                engine_keys[key].append(self.targets[iengine])
        # Convert to sorted list of tuples (key name, list of targets).
        keylist = []
        for key in sorted(engine_keys.keys()):
            targets = engine_keys[key]
            keylist.append((key, targets))
        return keylist

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

    def _create_local(self, local_call, shape, dtype, dist, grid_shape):
        """ Creates a local array, according to the method named in `local_call`."""
        shape_name, dtype_name, dist_name, grid_shape_name = self._key_and_push(shape, dtype, dist, grid_shape)
        da_key = self._generate_key()
        comm = self._comm_key
        cmd = '{da_key} = {local_call}({shape_name}, {dtype_name}, {dist_name}, {grid_shape_name}, {comm})'
        self._execute(cmd.format(**locals()))
        return DistArray.from_localarrays(da_key, self)

    def zeros(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        return self._create_local(local_call='distarray.local.zeros',
                                  shape=shape, dtype=dtype,
                                  dist=dist, grid_shape=grid_shape)

    def ones(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        return self._create_local(local_call='distarray.local.ones',
                                  shape=shape, dtype=dtype,
                                  dist=dist, grid_shape=grid_shape)

    def empty(self, shape, dtype=float, dist={0:'b'}, grid_shape=None):
        return self._create_local(local_call='distarray.local.empty',
                                  shape=shape, dtype=dtype,
                                  dist=dist, grid_shape=grid_shape)

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

        Block
        ~~~~~

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

        Cyclic
        ~~~~~~

        * ``dist_type`` is ``'c'``

        * ``proc_grid_size`` integer, greater than or equal to one.

        The size of the process grid in this dimension.  Equivalent to the
        number of local arrays in this dimension and determines the number of
        array sections.

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.

        * ``block_size`` integer, optional.  Greater than or equal to one.

        If not present, equivalent to being present with value of one.

        Unstructured
        ~~~~~~~~~~~~

        * ``dist_type`` is ``'u'``

        * ``indices`` sequence of one-dimensional numpy integer arrays or
          buffers.

          The ``len(indices)`` is the number of local unstructured arrays in
          this dimension.

          To compute the global size of the array in this dimension, compute
          ``sum(len(ii) for ii in indices)``.

        Not-distributed
        ~~~~~~~~~~~~~~~

        The ``'n'`` distribution type is a convenience to specify that an array
        is not distributed along this dimension.

        * ``dist_type`` is ``'n'``

        * ``size`` integer, greater than or equal to zero.

        The global size of the array in this dimension.

        """
        # global_dim_data is a sequence of dictionaries, one per dimension.
        mdmap = ClientMDMap.from_global_dim_data(self, global_dim_data)
        dim_data_per_rank = mdmap.get_local_dim_datas()
        return self._from_dim_data(dim_data_per_rank, dtype=dtype)

    def _from_dim_data(self, dim_data_per_rank, dtype=float):
        if len(self.targets) != len(dim_data_per_rank):
            errmsg = "`dim_data_per_rank` must contain a dim_data for every rank."
            raise TypeError(errmsg)

        da_key = self._generate_key()
        subs = ((da_key,) + self._key_and_push(dim_data_per_rank) +
                (self._comm_key,) + self._key_and_push(dtype) + (self._comm_key,))

        cmd = ('%s = distarray.local.LocalArray.'
               'from_dim_data(%s[%s.Get_rank()], dtype=%s, comm=%s)')
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
        subs = (da_key, name, self._comm_key)

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

    def load_npy(self, filename, dim_data_per_rank, grid_shape=None):
        """
        Load a DistArray from a dataset in a ``.npy`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        dim_data_per_rank : sequence of tuples of dict
            A "dim_data" data structure for every rank.  Described here:
            https://github.com/enthought/distributed-array-protocol
        grid_shape : tuple of int, optional
            Shape of process grid.

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

    def load_hdf5(self, filename, dim_data_per_rank, key='buffer',
                  grid_shape=None):
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
        grid_shape : tuple of int, optional
            Shape of process grid.

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

    def fromndarray(self, arr, dist={0: 'b'}, grid_shape=None):
        """Convert an ndarray to a distarray."""
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

    def close(self):
        self.client.close()
