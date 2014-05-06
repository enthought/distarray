# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------

"""
`Context` objects contain the information required for `DistArray`\s to
communicate with `LocalArray`\s.
"""

from __future__ import absolute_import

import collections
import atexit

import numpy

from distarray.dist import cleanup
from distarray.externals import six
from distarray.dist.distarray import DistArray
from distarray.dist.maps import Distribution

from distarray.dist.ipython_utils import IPythonClient
from distarray.utils import uid, DISTARRAY_BASE_NAME


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
        self.view.execute("from functools import reduce; "
                          "import distarray.local; "
                          "import distarray.local.mpiutils; "
                          "import distarray.utils; "
                          "import numpy")

        self.context_key = self._setup_context_key()
        self._comm_key = self._make_intracomm()
        self._set_engine_rank_mapping()

    def _setup_context_key(self):
        """
        Create a dict on the engines which will hold everything from
        this context.
        """
        context_key = uid()
        cmd = ("import types, sys;"
               "%s = types.ModuleType('%s');")
        cmd %= (context_key, context_key)
        self._execute(cmd, targets=range(len(self.view)))
        return context_key

    def _make_intracomm(self):
        def get_rank():
            from distarray.local.mpiutils import COMM_PRIVATE
            return COMM_PRIVATE.Get_rank()

        # self.view's engines must encompass all ranks in the MPI communicator,
        # i.e., everything in rank_map.values().
        def get_size():
            from distarray.local.mpiutils import COMM_PRIVATE
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
        cmd = "%s = distarray.local.mpiutils.create_comm_with_list(%s)"
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

    def _generate_key(self):
        """ Generate a unique key name for this context. """
        key = "%s.%s" % (self.context_key, 'key_' + uid())
        return key

    def _key_and_push(self, *values):
        keys = [self._generate_key() for value in values]
        self._push(dict(zip(keys, values)))
        return tuple(keys)

    def delete_key(self, key):
        """ Delete the specific key from all the engines. """
        cmd = ('try: del %s\n'
               'except NameError: pass') % key
        self._execute(cmd)

    def cleanup(self):
        """ Delete keys that this context created from all the engines. """
        cleanup.cleanup(view=self.view, module_name='__main__', prefix=self.context_key)

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

    def _create_local(self, local_call, distribution, dtype):
        """Creates LocalArrays with the method named in `local_call`."""
        da_key = self._generate_key()
        comm_name = self._comm_key
        ddpr = distribution.get_dim_data_per_rank()
        ddpr_name, dtype_name =  self._key_and_push(ddpr, dtype)
        cmd = ('{da_key} = {local_call}(distarray.local.maps.Distribution('
               '{ddpr_name}[{comm_name}.Get_rank()], comm={comm_name}), '
               'dtype={dtype_name})')
        self._execute(cmd.format(**locals()))
        return DistArray.from_localarrays(da_key, distribution=distribution,
                                          dtype=dtype)

    def empty(self, distribution, dtype=float):
        """Create an empty Distarray.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, with uninitialized values.
        """
        return self._create_local(local_call='distarray.local.empty',
                                  distribution=distribution, dtype=dtype)

    def zeros(self, distribution, dtype=float):
        """Create a Distarray filled with zeros.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with zeros.
        """
        return self._create_local(local_call='distarray.local.zeros',
                                  distribution=distribution, dtype=dtype)

    def ones(self, distribution, dtype=float):
        """Create a Distarray filled with ones.

        Parameters
        ----------
        distribution : Distribution object
        dtype : NumPy dtype, optional (default float)

        Returns
        -------
        DistArray
            A DistArray distributed as specified, filled with ones.
        """
        return self._create_local(local_call='distarray.local.ones',
                                  distribution=distribution, dtype=dtype,)

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

        return DistArray.from_localarrays(da_key, context=self)

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

    def load_npy(self, filename, distribution):
        """
        Load a DistArray from a dataset in a ``.npy`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        distribution: Distribution object

        Returns
        -------
        result : DistArray
            A DistArray encapsulating the file loaded.

        """
        da_key = self._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        subs = ((da_key,) + self._key_and_push(filename, ddpr) +
                (self._comm_key,) + (self._comm_key,))

        self._execute(
            '%s = distarray.local.load_npy(%s, %s[%s.Get_rank()], %s)' % subs
        )
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def load_hdf5(self, filename, distribution, key='buffer'):
        """
        Load a DistArray from a dataset in an ``.hdf5`` file.

        Parameters
        ----------
        filename : str
            Filename to load.
        distribution: Distribution object
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

        da_key = self._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        subs = ((da_key,) + self._key_and_push(filename, ddpr) +
                (self._comm_key,) + self._key_and_push(key) + (self._comm_key,))

        self._execute(
            '%s = distarray.local.load_hdf5(%s, %s[%s.Get_rank()], %s, %s)' % subs
        )
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def fromndarray(self, arr, distribution=None):
        """Create a DistArray from an ndarray.

        Parameters
        ----------
        distribution : Distribution object, optional
            If a Distribution object is not provided, one is created with
            `Distribution.from_shape(arr.shape)`.

        Returns
        -------
        DistArray
            A DistArray distributed as specified, using the values and dtype
            from `arr`.
        """
        if distribution is None:
            distribution = Distribution.from_shape(self, arr.shape)
        out = self.empty(distribution, dtype=arr.dtype)
        for index, value in numpy.ndenumerate(arr):
            out[index] = value
        return out

    fromarray = fromndarray

    def fromfunction(self, function, shape, **kwargs):
        """Create a DistArray from a function over global indices.

        Unlike numpy's `fromfunction`, the result of distarray's
        `fromfunction` is restricted to the same Distribution as the
        index array generated from `shape`.

        See numpy.fromfunction for more details.
        """
        dtype = kwargs.get('dtype', None)
        dist = kwargs.get('dist', None)
        grid_shape = kwargs.get('grid_shape', None)
        distribution = Distribution.from_shape(context=self,
                                               shape=shape, dist=dist,
                                               grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        function_name, ddpr_name, kwargs_name = \
            self._key_and_push(function, ddpr, kwargs)
        da_name = self._generate_key()
        comm_name = self._comm_key
        cmd = ('{da_name} = distarray.local.fromfunction({function_name}, '
               'distarray.local.maps.Distribution('
               '{ddpr_name}[{comm_name}.Get_rank()], comm={comm_name}),'
               '**{kwargs_name})')
        self._execute(cmd.format(**locals()))
        return DistArray.from_localarrays(da_name, distribution=distribution)

    def apply(self, func, args=None, kwargs=None, targets=None,
              result_name=None):
        """
        Analogous to IPython.parallel.view.apply_sync

        Parameters
        ----------
        func : function
        args : tuple
            positional arguments to func
        kwargs : dict
            key word arguments to func
        targets : sequence of integers
            engines func is to be run on.
        result_name : str
            The name given the result on the engines. If given this is returned
            to act as a proxy object.

        Returns
        -------
        if result_name is not None : str
            Name of the result on the engines.
        else: list
            A list of the results on all the engines.
        """

        def func_wrapper(func, result_name, args, kwargs):
            """
            Function which calls the applied function after grabbing all the
            arguments on the engines that are passed in as names of the form
            `__distarray__<some uuid>`.
            """
            main = __import__('__main__')
            prefix = main.distarray.utils.DISTARRAY_BASE_NAME

            # convert args
            args = list(args)
            for i, a in enumerate(args):
                if (isinstance(a, str) and a.startswith(prefix)):
                    args[i] = main.reduce(getattr, [main] + a.split('.'))
            args = tuple(args)

            # convert kwargs
            for k in kwargs.keys():
                val = kwargs[k]
                if (isinstance(val, str) and val.startswith(prefix)):
                    kwargs[k] = main.reduce(getattr, [main] + val.split('.'))

            if result_name:
                setattr(main, result_name, func(*args, **kwargs))
                return result_name
            else:
                return func(*args, **kwargs)

        # default arguments
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        wrapped_args = (func, result_name, args, kwargs)

        targets = self.targets if targets is None else targets

        result = self.view._really_apply(func_wrapper, args=wrapped_args,
                                          targets=targets, block=True)
        if result_name is not None:
            # result is a list of the same name 4 times, so just return 1.
            return result[0]
        else:
            return result
