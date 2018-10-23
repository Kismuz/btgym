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

from .base import BTgymBaseData, DataSampleConfig, EnvResetConfig
from .derivative import BTgymDataTrial, BTgymEpisode


def null_generator(num_points=10, **kwargs):
    """
    Dummy generator class.

    Args:
        num_points: trajectory length

    Returns:
        1d array of uniform randoms in [0,1]
    """
    return np.random.random(num_points)


class BaseDataGenerator():
    """
    Base synthetic data provider class.
    """
    def __init__(
            self,
            episode_duration=None,
            timeframe=1,
            generator_fn=null_generator,
            generator_params=None,
            name='BaseSyntheticDataGenerator',
            data_names=('default_asset',),
            global_time=None,
            task=0,
            log_level=WARNING,
            _nested_class_ref=None,
            _nested_params=None,
            **kwargs
    ):
        """

        Args:
            episode_duration:       dict, duration of episode in days/hours/mins
            generator_fn            callabale, should return generated data as 1D np.array
            generator_params        dict,
            timeframe:              int, data periodicity in minutes
            name:                   str
            data_names:             iterable of str
            global_time:            dict {y, m, d} to set custom global time (only for plotting)
            task:                   int
            log_level:              logbook.Logger level
            **kwargs:

        """
        # Logging:
        self.log_level = log_level
        self.task = task
        self.name = name
        self.filename = self.name + '_sample'

        self.data_names = data_names
        self.data_name = self.data_names[0]
        self.sample_instance = None
        self.metadata = {'sample_num': 0, 'type': None, 'parent_sample_type': None}

        self.data = None
        self.data_stat = None

        self.sample_num = 0
        self.is_ready = False

        if _nested_class_ref is None:
            self.nested_class_ref = BaseDataGenerator
        else:
            self.nested_class_ref = _nested_class_ref

        if _nested_params is None:
            self.nested_params = dict(
                episode_duration=episode_duration,
                timeframe=timeframe,
                generator_fn=generator_fn,
                generator_params=generator_params,
                name=name,
                data_names=data_names,
                task=task,
                log_level=log_level,
                _nested_class_ref=_nested_class_ref,
                _nested_params=_nested_params,
            )
        else:
            self.nested_params = _nested_params

        StreamHandler(sys.stdout).push_application()
        self.log = Logger('{}_{}'.format(self.name, self.task), level=self.log_level)

        # Default sample time duration:
        if episode_duration is None:
            self.episode_duration = dict(
                    days=0,
                    hours=23,
                    minutes=55,
                )
        else:
            self.episode_duration = episode_duration

        # Btfeed parsing setup:
        self.timeframe = timeframe
        self.names=['open']
        self.datetime = 0
        self.open = 1
        self.high = -1
        self.low = -1
        self.close = -1
        self.volume = -1
        self.openinterest = -1

        # base data feed related:
        self.params = {}
        if global_time is None:
            self.global_time = datetime.datetime(year=2018, month=1, day=1)
        else:
            self.global_time = datetime.datetime(**global_time)

        self.global_timestamp = self.global_time.timestamp()

        # Infer time indexes and sample number of records:
        self.train_index = pd.timedelta_range(
            start=datetime.timedelta(days=0, hours=0, minutes=0),
            end=datetime.timedelta(**self.episode_duration),
            freq='{}min'.format(self.timeframe)
        )
        self.test_index = pd.timedelta_range(
            start=self.train_index[-1] + datetime.timedelta(minutes=self.timeframe),
            periods=len(self.train_index),
            freq='{}min'.format(self.timeframe)
        )
        self.train_index += self.global_time
        self.test_index += self.global_time
        self.episode_num_records = len(self.train_index)

        self.generator_fn = generator_fn

        if generator_params is None:
            self.generator_params = {}

        else:
            self.generator_params = generator_params

    def set_logger(self, level=None, task=None):
        """
        Sets logbook logger.

        Args:
            level:  logbook.level, int
            task:   task id, int

        """
        if task is not None:
            self.task = task

        if level is not None:
            self.log = Logger('{}_{}'.format(self.name, self.task), level=level)

    def reset(self,  **kwargs):
        self.sample_num = 0
        self.is_ready = True

    def read_csv(self, **kwargs):
        self.data = self.generate_data(self.generator_params)

    def generate_data(self, generator_params, sample_type=0):
        """
        Generates data trajectory (episode)

        Args:
            generator_params:       dict, data generating parmeters
            sample_type:            0 - generate train data | 1 - generate test data

        Returns:
            data as pandas dataframe
        """
        assert sample_type in [0, 1], 'Expected sample type be either 0 (train), or 1 (test) got: {}'.format(sample_type)
        # Generate datapoints:
        data_array = self.generator_fn(num_points=self.episode_num_records, **generator_params)
        assert len(data_array.shape) == 1 and data_array.shape[0] == self.episode_num_records,\
            'Expected generated data to be 1D array of length {},  got data shape: {}'.format(
                self.episode_num_records,
                data_array.shape
            )
        negs = data_array[data_array < 0]
        if negs.any():
            self.log.warning(' Set to zero {} negative generated values'.format(negs.shape[0]))
            data_array[data_array < 0] = 0.0
        # Make dataframe:
        if sample_type:
            index = self.test_index
        else:
            index = self.train_index

        # data_dict = {name: data_array for name in self.names}
        # data_dict['hh:mm:ss'] = index
        df = pd.DataFrame(data={name: data_array for name in self.names}, index=index)
        # df = df.set_index('hh:mm:ss')
        return df

    def sample(self, get_new=True, sample_type=0,  **kwargs):
        """
        Samples continuous subset of data.

        Args:
            get_new (bool):                 not used;
            sample_type (int or bool):      0 (train) or 1 (test) - get sample from train or test data subsets
                                            respectively.

        Returns:
            Dataset instance with number of records ~ max_episode_len,

        """
        try:
            assert sample_type in [0, 1]

        except AssertionError:
            self.log.exception(
                'Sampling attempt: expected sample type be in {}, got: {}'.\
                format([0, 1], sample_type)
            )
            raise AssertionError

        if self.metadata['type'] is not None:
            if self.metadata['type'] != sample_type:
                self.log.warning(
                    'Attempted to sample type {} given current sample type {}, overriden.'.format(
                        self.metadata['type'],
                        sample_type
                    )
                )
                sample_type = self.metadata['type']

        # Generate data:
        sampled_data = self.generate_data(self.generator_params, sample_type=sample_type)
        self.sample_instance = self.nested_class_ref(**self.nested_params)
        self.sample_instance.filename += '_{}'.format(self.sample_num)
        self.log.info('New sample id: <{}>.'.format(self.sample_instance.filename))
        self.sample_instance.data = sampled_data

        # Add_metadata
        self.sample_instance.metadata['type'] = 'synthetic_data_sample'
        self.sample_instance.metadata['first_row'] = 0
        self.sample_instance.metadata['last_row'] = self.episode_num_records
        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = self.metadata['sample_num']
        self.sample_instance.metadata['parent_sample_type'] = self.metadata['type']
        self.sample_num += 1

        return self.sample_instance

    def describe(self):
        """
        Returns summary dataset statistic as pandas dataframe:

            - records count,
            - data mean,
            - data std dev,
            - min value,
            - 25% percentile,
            - 50% percentile,
            - 75% percentile,
            - max value

        for every data column.
        """
        # Pretty straightforward, using standard pandas utility.
        # The only caveat here is that if actual data has not been loaded yet, need to load, describe and unload again,
        # thus avoiding passing big files to BT server:
        flush_data = False
        try:
            assert not self.data.empty
            pass

        except (AssertionError, AttributeError) as e:
            self.read_csv()
            flush_data = True

        self.data_stat = self.data.describe()
        self.log.info('Data summary:\n{}'.format(self.data_stat.to_string()))

        if flush_data:
            self.data = None
            self.log.info('Flushed data.')

        return self.data_stat

    def to_btfeed(self):
        """
        Performs BTgymData-->bt.feed conversion.

        Returns:
             dict of type: {data_line_name: bt.datafeed instance}.
        """
        try:
            assert not self.data.empty
            btfeed = btfeeds.PandasDirectData(
                dataname=self.data,
                timeframe=self.timeframe,
                datetime=self.datetime,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
                openinterest=self.openinterest
            )
            btfeed.numrecords = self.data.shape[0]
            return {self.data_name: btfeed}

        except (AssertionError, AttributeError) as e:
            msg = 'Instance holds no data. Hint: forgot to call .read_csv()?'
            self.log.error(msg)
            raise AssertionError(msg)

    def set_global_timestamp(self, timestamp):
        pass


class SimpleTestTrial(BTgymDataTrial):
    """
    Truncated Trial without test period: always samples from train,
    sampled episode inherits tarin/test metadata of parent trail.
    """
    def __init__(
            self,
            filename=None,
            parsing_params=None,
            sampling_params=None,
            name=None,
            data_names=('default_asset',),
            frozen_time_split=None,
            task=0,
            log_level=WARNING,
            _config_stack=None,
            **kwargs
    ):
        nested_config = dict(
            class_ref=BTgymEpisode,
            kwargs=dict(
                parsing_params=parsing_params,
                sampling_params=sampling_params,
                name=name,
                task=task,
                log_level=log_level,
                _config_stack=None,
            )
        )
        super(SimpleTestTrial, self).__init__(
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=sampling_params,
            name=name,
            data_names=data_names,
            frozen_time_split=frozen_time_split,
            task=task,
            log_level=log_level,
            _config_stack=[nested_config],
        )

    def sample(self, sample_type=0, **kwargs):
        episode = self._sample(sample_type=0, **kwargs)
        episode.metadata['type'] = sample_type
        return episode


class CombinedDataGenerator(BaseDataGenerator):
    """
    Data provider class coupling synthetic train data and real test data.
    """
    def __init__(
            self,
            filename=None,
            parsing_params=None,
            episode_duration=None,
            time_gap=None,
            start_00=False,
            name='CombinedDataGenerator',
            **kwargs
    ):
        """

        Args:
            episode_duration:       dict, duration of episode in days/hours/mins
            generator_fn            callabale, should return generated data as 1D np.array
            generator_params        dict,
            timeframe:              int, data periodicity in minutes
            name:                   str
            data_names:             iterable of str
            global_time:            dict {y, m, d} to set custom global time (only for plotting)
            task:                   int
            log_level:              logbook.Logger level
            **kwargs:

        """
        super(CombinedDataGenerator, self).__init__(episode_duration=episode_duration, name=name, **kwargs)
        self.nested_params_test = dict(
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=dict(
                sample_duration=episode_duration,
                time_gap=time_gap,
                start_00=start_00,
                test_period={'days': 0, 'hours': 0, 'minutes': 0},
            ),

        )
        self.nested_params_test.update(self.nested_params)
        self.nested_class_ref_test = SimpleTestTrial

    def sample(self, get_new=True, sample_type=0,  **kwargs):
        """
        Samples continuous subset of data.

        Args:
            get_new (bool):                 not used;
            sample_type (int or bool):      0 (train) or 1 (test) - get sample from train or test data subsets
                                            respectively.

        Returns:
            Dataset instance with number of records ~ max_episode_len,

        """
        try:
            assert sample_type in [0, 1]

        except AssertionError:
            self.log.exception(
                'Sampling attempt: expected sample type be in {}, got: {}'.\
                format([0, 1], sample_type)
            )
            raise AssertionError

        if self.metadata['type'] is not None:
            if self.metadata['type'] != sample_type:
                self.log.warning(
                    'Attempted to sample type {} given current sample type {}, overriden.'.format(
                        self.metadata['type'],
                        sample_type
                    )
                )
                sample_type = self.metadata['type']
        if sample_type:
            # Got test, need natural-born data:
            self.sample_instance = self.nested_class_ref_test(**self.nested_params_test)
            self.log.info('New test sample id: <{}>.'.format(self.sample_instance.filename))

        else:
            # Generate train data:
            sampled_data = self.generate_data(self.generator_params, sample_type=sample_type)
            self.sample_instance = self.nested_class_ref(**self.nested_params)
            self.sample_instance.filename += '_{}'.format(self.sample_num)
            self.log.info('New train sample id: <{}>.'.format(self.sample_instance.filename))
            self.sample_instance.data = sampled_data

            # Add_metadata:
            #self.sample_instance.metadata['type'] = 'synthetic_data_sample'
            self.sample_instance.metadata['first_row'] = 0
            self.sample_instance.metadata['last_row'] = self.episode_num_records

        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = self.metadata['sample_num']
        self.sample_instance.metadata['parent_sample_type'] = self.metadata['type']
        self.sample_num += 1

        return self.sample_instance
