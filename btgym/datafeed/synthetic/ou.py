###############################################################################
#
# Copyright (C) 2017, 2018 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

from logbook import Logger, StreamHandler, WARNING

import datetime
import random
from numpy.random import beta as random_beta
import copy
import os
import sys

import backtrader.feeds as btfeeds
import numpy as np
import pandas as pd

from btgym.datafeed.synthetic.base import BaseCombinedDataGenerator


def ornshtein_uhlenbeck_process_fn(num_points, mu=0, l=0.1, sigma=1.0, x0=0, dt=1):
    """
    Generates Ornshtein-Uhlenbeck process realisation trajectory.

    Args:
        num_points:     int, trajectory length
        mu:             float, mean;
        l:              float, lambda, mean reversion rate;
        sigma:          float, volatility;
        x0:             float, starting point;
        dt:             int, time increment;

    Returns:
        generated data as 1D np.array
    """

    n = num_points
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] * np.exp(-l * dt) + mu * (1 - np.exp(-l * dt)) + \
               sigma * ((1 - np.exp(-2 * l * dt)) / (2 * l)) ** .5 * np.random.normal(0, 1)

    return x


def ornshtein_uhlenbeck_uniform_parameters_fn(mu=0, l=0.1, sigma=1.0, x0=None, dt=1):
    """
    If either mu, l and/or sigma is set in form of [a, b] - uniformly randomly samples parameters value
    form given interval.

    Args:
        mu:             float or iterable of 2 floats, mean;
        l:              float or iterable of 2 floats, lambda, mean reversion rate;
        sigma:          float or iterable of 2 floats, volatility;
        x0:             float or iterable of 2 floats, starting point;
        dt:             not used | int, time increment;

    Returns:
        dictionary of sampled values
    """
    if type(l) in [int, float, np.float64]:
        l = [l, l]
    else:
        l = list(l)

    if type(sigma) in [int, float, np.float64]:
        sigma = [sigma, sigma]
    else:
        sigma = list(sigma)

    if type(mu) in [int, float, np.float64]:
        mu = [mu, mu]
    else:
        sigma = list(mu)

    # Sanity checks:
    assert len(l) == 2 and 0 < l[0] <= l[-1], \
        'Expected OU mean reversion rate be positive float or ordered interval, got: {}'.format(l)
    assert len(sigma) == 2 and 0 <= sigma[0] <= sigma[-1], \
        'Expected OU sigma be non-negative float or ordered interval, got: {}'.format(sigma)
    assert len(mu) == 2 and mu[0] <= mu[-1], \
        'Expected OU mu be float or ordered interval, got: {}'.format(mu)

    # Uniformly sample params:
    l_value = np.random.uniform(low=l[0], high=l[-1])
    sigma_value = np.random.uniform(low=sigma[0], high=sigma[-1])
    mu_value = np.random.uniform(low=mu[0], high=mu[-1])

    # Choose starting point equal to mean:
    if x0 is None:
        x0_value = mu_value

    else:
        x0_value = x0

    return dict(
        l=l_value,
        sigma=sigma_value,
        mu=mu_value,
        x0=x0_value,
        #dt=dt
    )


class UniformOUGenerator(BaseCombinedDataGenerator):
    """
    Combined data iterator provides:
    - realisations of Ornstein-Uhlenbeck process as train data;
    - real historic data as test data;

    OUp. paramters are randomly uniformly sampled from given intervals
    """
    @staticmethod
    def fix_args(fn, **kwargs):
        def f_empty():
            return fn(**kwargs)
        return f_empty

    def __init__(self, ou_mu=0, ou_lambda=0.1, ou_sigma=1, name='UniformOUData', **kwargs):
        """

        Args:
            ou_mu:                      float or iterable of 2 floats, Ornstein-Uhlenbeck process mean value or interval
            ou_lambda:                  float or iterable of 2 floats, OUp. mean-reverting rate or interval
            ou_sigma:                   float or iterable of 2 floats, OUp. volatility value or interval
            filename:                   str, test data filename
            parsing_params:             dict test data parsing params
            episode_duration_train:     dict, duration of train episode in days/hours/mins
            episode_duration_test:      dict, duration of test episode in days/hours/mins
            time_gap:                   dict test episode duration tolerance
            start_00:                   bool, def=False
            timeframe:                  int, data periodicity in minutes
            name:                       str
            data_names:                 iterable of str
            global_time:                dict {y, m, d} to set custom global time (here for plotting only)
            task:                       int
            log_level:                  logbook.Logger level
        """
        super(UniformOUGenerator, self).__init__(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=self.fix_args(
                ornshtein_uhlenbeck_uniform_parameters_fn,
                mu=ou_mu,
                l=ou_lambda,
                sigma=ou_sigma,
            ),
            name=name,
            **kwargs
        )


class OUGenerator(BaseCombinedDataGenerator):
    """
    Combined data iterator provides:
    - realisations of Ornstein-Uhlenbeck process as train data;
    - real historic data as test data;

    This class expects OUp. parameters no-args callable to be explicitly provided.
    """
    @staticmethod
    def fix_args(fn, **kwargs):
        def f_empty():
            return fn(**kwargs)
        return f_empty

    def __init__(self, generator_parameters_fn, name='OUData', **kwargs):
        """

        Args:
            generator_parameters_fn:    callable, should return dictionary of generator_fn kwargs: {l, mu, sigma};
                                        this callable itself should not require any args
            filename:                   str, test data filename
            parsing_params:             dict test data parsing params
            episode_duration_train:     dict, duration of train episode in days/hours/mins
            episode_duration_test:      dict, duration of test episode in days/hours/mins
            time_gap:                   dict test episode duration tolerance
            start_00:                   bool, def=False
            timeframe:                  int, data periodicity in minutes
            name:                       str
            data_names:                 iterable of str
            global_time:                dict {y, m, d} to set custom global time (here for plotting only)
            task:                       int
            log_level:                  logbook.Logger level
        """
        super(OUGenerator, self).__init__(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=generator_parameters_fn,
            name=name,
            **kwargs
        )




