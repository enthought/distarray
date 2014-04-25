# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------


"""Emulate numpy.random"""

from __future__ import absolute_import

from distarray.dist.distarray import DistArray
from distarray.dist.client_map import Distribution


class Random(object):

    def __init__(self, context):
        self.context = context
        self.context._execute('import distarray.local.random')

    def seed(self, seed=None):
        """
        Seed the random number generators on each engine.

        Parameters
        ----------
        seed : None, int, or array of integers
            Base random number seed to use on each engine.
            If None, then a non-deterministic seed is obtained from the
            operating system. Otherwise, the seed is used as passed,
            and the sequence of random numbers will be deterministic.

            Each individual engine has its state adjusted so that
            it is different from each other engine. Thus, each engine
            will compute a different sequence of random numbers.
        """
        cmd = 'numpy.random.seed(seed=%r)' % (seed)
        self.context._execute(cmd)
        cmd = 'distarray.local.random.label_state(%s)' % (
            self.context._comm_key)
        self.context._execute(cmd)

    def rand(self, size=None, dist=None, grid_shape=None):
        """
        rand(size=(d0, d1, ..., dn))

        Random values in a given shape.

        Create a distarray of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        size : tuple of ints
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.
        dist : dist dictionary
            Dictionary describing how to distribute the array along each axis.
        grid_shape : tuple
            Tuple describing the processor grid topology.

        Returns
        -------
        out : distarray, shape ``(d0, d1, ..., dn)``
            Random values.

        """
        if dist is None:
            dist = {0: 'b'}
        da_key = self.context._generate_key()

        distribution = Distribution.from_shape(context=self.context,
                                               shape=size,
                                               dist=dist,
                                               grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        ddpr_name = self.context._key_and_push(ddpr)[0]
        comm_name = self.context._comm_key
        self.context._execute(
            '{da_key} = distarray.local.random.rand('
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()))
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def normal(self, loc=0.0, scale=1.0, size=None, dist=None,
               grid_shape=None):
        """
        normal(loc=0.0, scale=1.0, size=None, dist={0: 'b'}, grid_shape=None)

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float
            Mean ("centre") of the distribution.
        scale : float
            Standard deviation (spread or "width") of the distribution.
        size : tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.
        dist : dist dictionary
            Dictionary describing how to distribute the array along each axis.
        grid_shape : tuple
            Tuple describing the processor grid topology.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation.  The square of the standard deviation, :math:`\\sigma^2`, is
        called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        `numpy.random.normal` is more likely to return samples lying close to
        the mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               http://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        """
        if dist is None:
            dist = {0: 'b'}
        da_key = self.context._generate_key()

        distribution = Distribution.from_shape(context=self.context,
                                               shape=size,
                                               dist=dist,
                                               grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        loc_name, scale_name, ddpr_name = \
            self.context._key_and_push(loc, scale, ddpr)
        comm_name = self.context._comm_key
        self.context._execute(
            '{da_key} = distarray.local.random.normal('
            'loc={loc_name}, scale={scale_name},'
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()))
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def randint(self, low, high=None, size=None, dist=None, grid_shape=None):
        """
        randint(low, high=None, size=None)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            if provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape. Default is None, in which case a single int is
            returned.
        dist : dist dictionary
            Dictionary describing how to distribute the array along each axis.
        grid_shape : tuple
            Tuple describing the processor grid topology.

        Returns
        -------
        out : distarray of ints
            `size`-shaped distarray of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        """
        if dist is None:
            dist = {0: 'b'}
        da_key = self.context._generate_key()

        distribution = Distribution.from_shape(context=self.context,
                                               shape=size,
                                               dist=dist,
                                               grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        low_name, high_name, ddpr_name = \
            self.context._key_and_push(low, high, ddpr)
        comm_name = self.context._comm_key
        self.context._execute(
            '{da_key} = distarray.local.random.randint('
            'low={low_name}, high={high_name},'
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()))
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def randn(self, size=None, dist=None, grid_shape=None):
        """
        randn(size=(d0, d1, ..., dn))

        Return samples from the "standard normal" distribution.

        Parameters
        ----------
        size : tuple of ints
            The dimensions of the returned array, should be all positive.
            If no argument is given a single Python float is returned.
        dist : dist dictionary
            Dictionary describing how to distribute the array along each axis.
        grid_shape : tuple
            Tuple describing the processor grid topology.

        Returns
        -------
        out : distarray
            A ``(d0, d1, ..., dn)``-shaped distarray of floating-point samples
            from the standard normal distribution.

        """
        if dist is None:
            dist = {0: 'b'}
        da_key = self.context._generate_key()

        distribution = Distribution.from_shape(context=self.context,
                                               shape=size,
                                               dist=dist,
                                               grid_shape=grid_shape)
        ddpr = distribution.get_dim_data_per_rank()
        ddpr_name = self.context._key_and_push(ddpr)[0]
        comm_name = self.context._comm_key
        self.context._execute(
            '{da_key} = distarray.local.random.randn('
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()))
        return DistArray.from_localarrays(da_key, distribution=distribution)
