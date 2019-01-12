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
import sys, os
import copy

import backtrader.feeds as btfeeds
import numpy as np
import pandas as pd

from btgym.datafeed.derivative import BTgymDataset2
from btgym.datafeed.multi import BTgymMultiData


def base_random_generator_fn(num_points=10, **kwargs):
    """
    Base random uniform generating function. Provides synthetic data points.

    Args:
        num_points: trajectory length
        kwargs:     any function parameters, not used here

    Returns:
        1d array of generated values; here: randoms in [0,1]
    """
    return np.random.random(num_points)


def base_bias_generator_fn(num_points=10, bias=1, **kwargs):
    """
    Base bias generating function. Provides constant synthetic data points.

    Args:
        num_points: trajectory length
        bias:       data point constant value >=0
        kwargs:     any function parameters, not used here

    Returns:
        1d array of generated values; here: randoms in [0,1]
    """
    assert bias >= 0, 'Only positive bias allowed, got: {}'.format(bias)
    return np.ones(num_points) * bias


def base_generator_parameters_fn(**kwargs):
    """
    Base parameters generating function. Provides arguments for data generating function.
    It itself accept arguments specified via `generator_parameters_config` dictionary;

    Returns:
        dictionary of kwargs consistent with generating function used.
    """
    return dict()


def base_random_uniform_parameters_fn(**kwargs):
    """
    Provides samples for kwargs given.
    If parameter is set as float - returns exactly given value;
    if parameter is set as iterable of form [a, b] - uniformly randomly samples parameters value
    form given interval.

    Args:
        **kwargs: any kwarg specifying float or iterable of two ordered floats

    Returns:
        dictionary of kwargs holding sampled values
    """
    samples = {}
    for key, value in kwargs.items():
        if type(value) in [int, float, np.float64]:
            interval = [value, value]
        else:
            interval = list(value)

        assert len(interval) == 2 and interval[0] <= interval[-1], \
            'Expected parameter <{}> be float or ordered interval, got: {}'.format(key, value)

        samples[key] = np.random.uniform(low=interval[0], high=interval[-1])
    return samples


def base_spread_generator_fn(num_points=10, alpha=1, beta=1, minimum=0, maximum=0):
    """
    Generates spread values for single synthetic tragectory. Samples drawn from parametrized beta-distribution;
    If base generated trajectory P is given, than High/Ask value = P + 1/2 * Spread; Low/Bid value = P - 1/2* Spread

    Args:
        num_points: trajectory length
        alpha:      beta-distribution alpha param.
        beta:       beta-distribution beta param.
        minimum:    spread minimum value
        maximum:    spread maximum value

    Returns:
        1d array of generated values;
    """
    assert alpha > 0 and beta > 0, 'Beta-distribution parameters should be non-negative, got: {},{}'.format(alpha, beta)
    assert minimum <= maximum, 'Spread min/max values should form ordered pair, got: {}/{}'.format(minimum, maximum)
    return minimum + np.random.beta(a=alpha, b=beta, size=num_points) * (maximum - minimum)


class BaseDataGenerator:
    """
    Base synthetic data provider class.
    """
    def __init__(
            self,
            episode_duration=None,
            timeframe=1,
            generator_fn=base_random_generator_fn,
            generator_parameters_fn=base_generator_parameters_fn,
            generator_parameters_config=None,
            spread_generator_fn=None,
            spread_generator_parameters=None,
            name='BaseSyntheticDataGenerator',
            data_names=('default_asset',),
            parsing_params=None,
            target_period=-1,
            global_time=None,
            task=0,
            log_level=WARNING,
            _nested_class_ref=None,
            _nested_params=None,
            **kwargs
    ):
        """

        Args:
            episode_duration:               dict, duration of episode in days/hours/mins
            generator_fn                    callabale, should return generated data as 1D np.array
            generator_parameters_fn:        callable, should return dictionary of generator_fn kwargs
            generator_parameters_config:    dict, generator_parameters_fn args
            spread_generator_fn:            callable, should return values of spread to form {High, Low}
            spread_generator_parameters:    dict, spread_generator_fn args
            timeframe:                      int, data periodicity in minutes
            name:                           str
            data_names:                     iterable of str
            target_period:                  int or dict, if set to -1 - disables `test` sampling
            global_time:                    dict {y, m, d} to set custom global time (only for plotting)
            task:                           int
            log_level:                      logbook.Logger level
            **kwargs:

        """
        # Logging:
        self.log_level = log_level
        self.task = task
        self.name = name
        self.filename = self.name + '_sample'
        self.target_period = target_period

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
                generator_parameters_fn=generator_parameters_fn,
                generator_parameters_config=generator_parameters_config,
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
        if parsing_params is None:
            self.parsing_params = dict(
                names=['ask', 'bid', 'mid'],
                datetime=0,
                timeframe=1,
                open='mid',
                high='ask',
                low='bid',
                close='mid',
                volume=-1,
                openinterest=-1
            )
        else:
            self.parsing_params = parsing_params

        self.columns_map = {
            'open': 'mean',
            'high': 'maximum',
            'low': 'minimum',
            'close': 'mean',
            'bid': 'minimum',
            'ask': 'maximum',
            'mid': 'mean',
            'volume': 'nothing',
        }
        self.nested_params['parsing_params'] = self.parsing_params

        for key, value in self.parsing_params.items():
            setattr(self, key, value)

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
        self.generator_parameters_fn = generator_parameters_fn

        if generator_parameters_config is not None:
            self.generator_parameters_config = generator_parameters_config

        else:
            self.generator_parameters_config = {}

        self.spread_generator_fn = spread_generator_fn

        if spread_generator_parameters is not None:
            self.spread_generator_parameters = spread_generator_parameters

        else:
            self.spread_generator_parameters = {}

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
        self.read_csv()
        self.sample_num = 0
        self.is_ready = True

    def read_csv(self, **kwargs):
        self.data = self.generate_data(self.generator_parameters_fn(**self.generator_parameters_config))

    def generate_data(self, generator_params, sample_type=0):
        """
        Generates data trajectory, performs base consistency checks.

        Args:
            generator_params:       dict, data_generating_function parameters
            sample_type:            0 - generate train data | 1 - generate test data

        Returns:
            data as pandas dataframe
        """
        assert sample_type in [0, 1],\
            'Expected sample type be either 0 (train), or 1 (test) got: {}'.format(sample_type)
        # Generate data points:
        data_array = self.generator_fn(num_points=self.episode_num_records, **generator_params)

        assert len(data_array.shape) == 1 and data_array.shape[0] == self.episode_num_records,\
            'Expected generated data to be 1D array of length {},  got data shape: {}'.format(
                self.episode_num_records,
                data_array.shape
            )
        if self.spread_generator_fn is not None:
            spread_array = self.spread_generator_fn(
                num_points=self.episode_num_records,
                **self.spread_generator_parameters
            )
            assert len(spread_array.shape) == 1 and spread_array.shape[0] == self.episode_num_records, \
                'Expected generated spread to be 1D array of length {},  got data shape: {}'.format(
                    self.episode_num_records,
                    spread_array.shape
                )
        else:
            spread_array = np.zeros(self.episode_num_records)

        data_dict = {
            'mean': data_array,
            'maximum': data_array + .5 * spread_array,
            'minimum': data_array - .5 * spread_array,
            'nothing': data_array * 0.0,
        }

        # negs = data_dict['minimum'] < 0
        # if negs.any():
        #     self.log.warning('{} negative generated values detected'.format(negs.shape[0]))

        # Make dataframe:
        if sample_type:
            index = self.test_index
        else:
            index = self.train_index
        # Map dictionary of data to dataframe columns:
        df = pd.DataFrame(data={name: data_dict[self.columns_map[name]] for name in self.names}, index=index)
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
            Dataset instance with number of records ~ max_episode_len.

        """
        try:
            assert sample_type in [0, 1]

        except AssertionError:
            msg = 'Sampling attempt: expected sample type be in {}, got: {}'.format([0, 1], sample_type)
            self.log.error(msg)
            raise ValueError(msg)

        if self.target_period == -1 and sample_type:
            msg = 'Attempt to sample type {} given disabled target_period'.format(sample_type)
            self.log.error(msg)
            raise ValueError(msg)

        if self.metadata['type'] is not None:
            if self.metadata['type'] != sample_type:
                self.log.warning(
                    'Attempt to sample type {} given current sample type {}, overriden.'.format(
                        sample_type,
                        self.metadata['type']
                    )
                )
                sample_type = self.metadata['type']

        # Get sample:
        self.sample_instance = self.sample_synthetic(sample_type)

        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = self.metadata['sample_num']
        self.sample_instance.metadata['parent_sample_type'] = self.metadata['type']
        self.sample_num += 1

        return self.sample_instance

    def sample_synthetic(self, sample_type=0):
        """
        Get data_generator instance containing synthetic data.

        Args:
            sample_type (int or bool):      0 (train) or 1 (test) - get sample with train or test time periods
                                            respectively.

        Returns:
            nested_class_ref instance
        """
        # Generate data:
        generator_params = self.generator_parameters_fn(**self.generator_parameters_config)
        data = self.generate_data(generator_params, sample_type=sample_type)

        # Make data_class instance:
        sample_instance = self.nested_class_ref(**self.nested_params)
        sample_instance.filename += '_{}'.format(self.sample_num)
        self.log.info('New sample id: <{}>.'.format(sample_instance.filename))

        # Add data and metadata:
        sample_instance.data = data
        sample_instance.metadata['generator'] = generator_params
        sample_instance.metadata['first_row'] = 0
        sample_instance.metadata['last_row'] = self.episode_num_records

        return sample_instance

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


class BaseCombinedDataSet:
    """
    Data provider class wrapper incorporates synthetic train and real test data streams.
    """
    def __init__(
            self,
            train_data_config,
            test_data_config,
            train_class_ref=BaseDataGenerator,
            test_class_ref=BTgymDataset2,
            name='CombinedDataSet',
            **kwargs
    ):
        """

        Args:
            filename:                       str, test data filename
            parsing_params:                 dict test data parsing params
            episode_duration_train:         dict, duration of train episode in days/hours/mins
            episode_duration_test:          dict, duration of test episode in days/hours/mins
            time_gap:                       dict test episode duration tolerance
            start_00:                       bool, def=False
            generator_fn                    callabale, should return generated data as 1D np.array
            generator_parameters_fn:        callable, should return dictionary of generator_fn kwargs
            generator_parameters_config:    dict, generator_parameters_fn args
            timeframe:                      int, data periodicity in minutes
            name:                           str
            data_names:                     iterable of str
            global_time:                    dict {y, m, d} to set custom global time (here for plotting only)
            task:                           int
            log_level:                      logbook.Logger level
            **kwargs:                       common kwargs

        """
        self.name = name
        self.log = None

        try:
            self.task = kwargs['task']
        except KeyError:
            self.task = None

        self.train_data_config = train_data_config
        self.test_data_config = test_data_config
        self.train_data_config.update(kwargs)
        self.test_data_config.update(kwargs)
        self.train_data_config['name'] = self.name + '/train'
        self.test_data_config['name'] = self.name + '/test'

        # Declare all test data come from target domain:
        self.test_data_config['target_period'] = -1
        self.test_data_config['test_period'] = -1

        self.streams = {
            'train': train_class_ref(**self.train_data_config),
            'test': test_class_ref(**self.test_data_config),
        }

        self.sample_instance = None
        self.sample_num = 0
        self.is_ready = False

        # Legacy parameters, left here for BTgym API_shell:
        try:
            self.parsing_params = kwargs['parsing_params']

        except KeyError:
            self.parsing_params = dict(
                sep=',',
                header=0,
                index_col=0,
                parse_dates=True,
                names=['ask', 'bid', 'mid'],
                dataname=None,
                datetime=0,
                nullvalue=0.0,
                timeframe=1,
                high=1,  # 'ask',
                low=2,  # 'bid',
                open=3,  # 'mid',
                close=3,  # 'mid',
                volume=-1,
                openinterest=-1,
            )

        try:
            self.sampling_params = kwargs['sampling_params']

        except KeyError:
            self.sampling_params = {}

        self.params = {}
        self.params.update(self.parsing_params)
        self.params.update(self.sampling_params)

        self.set_params(self.params)
        self.data_names = self.streams['test'].data_names
        self.global_timestamp = 0

    def set_params(self, params_dict):
        """
        Batch attribute setter.

        Args:
            params_dict: dictionary of parameters to be set as instance attributes.
        """
        for key, value in params_dict.items():
            setattr(self, key, value)

    def set_logger(self, *args, **kwargs):
        for stream in self.streams.values():
            stream.set_logger(*args, **kwargs)
        self.log = self.streams['test'].log

    def reset(self, *args, **kwargs):
        for stream in self.streams.values():
            stream.reset(*args, **kwargs)
        self.task = self.streams['test'].task
        self.global_timestamp = self.streams['test'].global_timestamp
        self.sample_num = 0
        self.is_ready = True

    def read_csv(self, *args, **kwargs):
        for stream in self.streams.values():
            stream.read_csv(*args, **kwargs)

    def describe(self,*args, **kwargs):
        return self.streams['test'].describe()

    def set_global_timestamp(self, *args, **kwargs):
        for stream in self.streams.values():
            stream.set_global_timestamp(*args, **kwargs)
        self.global_timestamp = self.streams['test'].global_timestamp

    def to_btfeed(self):
        raise NotImplementedError

    def sample(self, sample_type=0,  **kwargs):
        """
        Samples continuous subset of data.

        Args:
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

        if sample_type:
            self.sample_instance = self.streams['test'].sample(sample_type=sample_type, **kwargs)
            self.sample_instance.metadata['generator'] = {}

        else:
            self.sample_instance = self.streams['train'].sample(sample_type=sample_type, **kwargs)

        # Common metadata:
        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = 0
        self.sample_instance.metadata['parent_sample_type'] = None
        self.sample_num += 1

        return self.sample_instance


class BasePairDataGenerator(BTgymMultiData):
    """
    Generates pair of data streams driven by single 2-level generating process.
    TODO: make data generating process single stand-along function or class method, do not use BaseDataGenerator's
    """
    def __init__(
            self,
            data_names,
            process1_config=None,  # bias generator
            process2_config=None,  # spread generator
            data_class_ref=BaseDataGenerator,
            name='PairDataGenerator',
            _top_level=True,
            **kwargs
    ):
        assert len(list(data_names)) == 2, 'Expected `data_names` be pair of `str`, got: {}'.format(data_names)
        if process1_config is None:
            self.process1_config = {
                'generator_fn': base_bias_generator_fn,
                'generator_parameters_fn': base_generator_parameters_fn,
                'generator_parameters_config': None,
            }
        else:
            self.process1_config = process1_config

        if process2_config is None:
            self.process2_config = {
                'generator_fn': base_random_generator_fn,
                'generator_parameters_fn': base_generator_parameters_fn,
                'generator_parameters_config': None,
            }
        else:
            self.process2_config = process2_config

        data_config = {name: {'filename': None, 'config': {}} for name in data_names}

        # Let first asset hold p1 generating process:
        self.a1_name = data_names[0]
        data_config[self.a1_name]['config'].update(self.process1_config)

        # Second asset will hold p2 generating process:
        self.a2_name = data_names[-1]
        data_config[self.a2_name]['config'].update(self.process2_config)

        self.nested_kwargs = kwargs
        self.get_new_sample = not _top_level

        super(BasePairDataGenerator, self).__init__(
            data_config=data_config,
            data_names=data_names,
            data_class_ref=data_class_ref,
            name=name,
            **kwargs
        )

    def sample(self, sample_type=0, **kwargs):
        if self.get_new_sample:
            # Get process1 trajectory:
            p1_sample = self.data[self.a1_name].sample(sample_type=sample_type, **kwargs)

            # Get p2 trajectory:
            p2_sample = self.data[self.a2_name].sample(sample_type=sample_type, **kwargs)

            idx_intersected = p1_sample.data.index.intersection(p2_sample.data.index)

            self.log.info('p1/p2 shared num. records: {}'.format(len(idx_intersected)))
            # TODO: move this generating process to stand-along function

            # Combine processes:
            data1 = p1_sample.data + 0.5 * p2_sample.data
            data2 = p1_sample.data - 0.5 * p2_sample.data
            metadata = copy.deepcopy(p2_sample.metadata)

        else:
            data1 = None
            data2 = None
            metadata = {}

        metadata.update(
            {'type': sample_type, 'sample_num': self.sample_num, 'parent_sample_type': self.sample_num, 'parent_sample_num': sample_type}
        )
        # Prepare empty instance of multi_stream data:
        sample = BasePairDataGenerator(
            data_names=self.data_names,
            process1_config=self.process1_config,
            process2_config=self.process2_config,
            data_class_ref=self.data_class_ref,
            # task=self.task,
            # log_level=self.log_level,
            name='sub_' + self.name,
            _top_level=False,
            **self.nested_kwargs
        )
        # TODO: maybe add p1 metadata
        sample.metadata = copy.deepcopy(metadata)

        # Populate sample with data:
        sample.data[self.a1_name].data = data1
        sample.data[self.a2_name].data = data2

        sample.filename = {key: stream.filename for key, stream in self.data.items()}
        self.sample_num += 1
        return sample


class BasePairCombinedDataSet(BaseCombinedDataSet):
    """
    Provides doubled streams of simulated train / real test data.
    Suited for pairs or spread trading setup.
    """
    def __init__(
            self,
            assets_filenames,
            process1_config=None,
            process2_config=None,
            train_episode_duration=None,
            test_episode_duration=None,
            train_class_ref=BasePairDataGenerator,
            test_class_ref=BTgymMultiData,
            name='PairCombinedDataSet',
            **kwargs
    ):
        assert isinstance(assets_filenames, dict),\
            'Expected `assets_filenames` type `dict`, got {} '.format(type(assets_filenames))

        data_names = [name for name in assets_filenames.keys()]
        assert len(data_names) == 2, 'Expected exactly two assets, got: {}'.format(data_names)

        train_data_config = dict(
            data_names=data_names,
            process1_config=process1_config,
            process2_config=process2_config,
            data_class_ref=BaseDataGenerator,
            episode_duration=train_episode_duration,
            # name=name,
        )
        test_data_config = dict(
            data_class_ref=BTgymDataset2,
            data_config={asset_name: {'filename': file_name} for asset_name, file_name in assets_filenames.items()},
            episode_duration=test_episode_duration,
            # name=name,
        )

        super(BasePairCombinedDataSet, self).__init__(
            train_data_config=train_data_config,
            test_data_config=test_data_config,
            train_class_ref=train_class_ref,
            test_class_ref=test_class_ref,
            name=name,
            **kwargs
        )

