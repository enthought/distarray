# encoding: utf-8
# ---------------------------------------------------------------------------
#  Copyright (C) 2008-2014, IPython Development Team and Enthought, Inc.
#  Distributed under the terms of the BSD License.  See COPYING.rst.
# ---------------------------------------------------------------------------


"""Emulate numpy.random"""

from __future__ import absolute_import

from distarray.dist.distarray import DistArray
from distarray.dist.maps import Distribution


class Random(object):

    def __init__(self, context):
        self.context = context
        self.context._execute('import distarray.local.random',
                              targets=self.context.targets)

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
        self.context._execute(cmd, targets=self.context.targets)
        cmd = 'distarray.local.random.label_state(%s)' % (
            self.context.comm)
        self.context._execute(cmd, targets=self.context.targets)

    def rand(self, distribution):
        """Random values over a given distribution.

        Create a distarray of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        distribution : Distribution object

        Returns
        -------
        out : DistArray
            Random values.

        """
        da_key = self.context._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        ddpr_name = self.context._key_and_push(ddpr)[0]
        comm_name = self.context.comm
        self.context._execute(
            '{da_key} = distarray.local.random.rand('
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()),
            targets=distribution.targets)
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def normal(self, distribution, loc=0.0, scale=1.0):
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
        distribution : Distribution object

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
        da_key = self.context._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        loc_name, scale_name, ddpr_name = \
            self.context._key_and_push(loc, scale, ddpr)
        comm_name = self.context.comm
        self.context._execute(
            '{da_key} = distarray.local.random.normal('
            'loc={loc_name}, scale={scale_name},'
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()),
            targets=distribution.targets)
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def randint(self, distribution, low, high=None):
        """Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).

        Parameters
        ----------
        distribution : Distribution object
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
        da_key = self.context._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        low_name, high_name, ddpr_name = \
            self.context._key_and_push(low, high, ddpr)
        comm_name = self.context.comm
        self.context._execute(
            '{da_key} = distarray.local.random.randint('
            'low={low_name}, high={high_name},'
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()),
            targets=distribution.targets)
        return DistArray.from_localarrays(da_key, distribution=distribution)

    def randn(self, distribution):
        """Return samples from the "standard normal" distribution.

        Parameters
        ----------
        distribution : Distribution object

        Returns
        -------
        out : DistArray
            A DistArray of floating-point samples from the standard normal
            distribution.
        """
        da_key = self.context._generate_key()
        ddpr = distribution.get_dim_data_per_rank()
        ddpr_name = self.context._key_and_push(ddpr)[0]
        comm_name = self.context.comm
        self.context._execute(
            '{da_key} = distarray.local.random.randn('
            'distribution=distarray.local.maps.Distribution('
            'dim_data={ddpr_name}[{comm_name}.Get_rank()], '
            'comm={comm_name}))'.format(**locals()),
            targets=distribution.targets)
        return DistArray.from_localarrays(da_key, distribution=distribution)
