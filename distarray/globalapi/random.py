# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------


"""
Contains the `Random` class that emulates `numpy.random` for `DistArray`.
"""

from __future__ import absolute_import

from distarray.globalapi.distarray import DistArray
from distarray.globalapi.maps import asdistribution


class Random(object):

    def __init__(self, context):
        self.context = context

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
        def _local_setup_random(seed, comm):
            from numpy import random
            from distarray.localapi.random import label_state
            random.seed(seed)
            label_state(comm)

        self.context.apply(_local_setup_random,
                           (seed, self.context.comm),
                           targets=self.context.targets)

    def rand(self, shape_or_dist):
        """Random values over a given distribution.

        Create a distarray of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object

        Returns
        -------
        out : DistArray
            Random values.

        """
        return self._local_rand_call('rand', shape_or_dist)

    def normal(self, shape_or_dist, loc=0.0, scale=1.0):
        """Draw random samples from a normal (Gaussian) distribution.

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
        shape_or_dist : shape tuple or Distribution object

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
        return self._local_rand_call('normal', shape_or_dist,
                                     dict(loc=loc, scale=scale))

    def randint(self, shape_or_dist, low, high=None):
        """Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            if provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).

        Returns
        -------
        out : DistArray of ints
            DistArray of random integers from the appropriate distribution.

        """
        return self._local_rand_call('randint', shape_or_dist,
                                     dict(low=low, high=high))

    def randn(self, shape_or_dist):
        """Return samples from the "standard normal" distribution.

        Parameters
        ----------
        shape_or_dist : shape tuple or Distribution object

        Returns
        -------
        out : DistArray
            A DistArray of floating-point samples from the standard normal
            distribution.
        """
        return self._local_rand_call('randn', shape_or_dist)

    def _local_rand_call(self, local_func_name, shape_or_dist, kwargs=None):

        kwargs = kwargs or {}

        def _local_call(comm, local_func_name, ddpr, kwargs):
            import distarray.localapi.random as local_random
            from distarray.localapi.maps import Distribution
            local_func = getattr(local_random, local_func_name)
            if len(ddpr):
                dim_data = ddpr[comm.Get_rank()]
            else:
                dim_data = ()
            dist = Distribution(dim_data=dim_data, comm=comm)
            return proxyize(local_func(distribution=dist, **kwargs))

        distribution = asdistribution(self.context, shape_or_dist)
        ddpr = distribution.get_dim_data_per_rank()
        args = (distribution.comm, local_func_name, ddpr, kwargs)

        da_key = self.context.apply(_local_call, args,
                                    targets=distribution.targets)
        return DistArray.from_localarrays(da_key[0], distribution=distribution)
