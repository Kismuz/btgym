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

from btgym.datafeed.derivative import BTgymDataTrial, BTgymEpisode
from btgym.datafeed.multi import BTgymMultiData


def base_generator_fn(num_points=10, **kwargs):
    """
    Base generating function. Provides synthetic data points.

    Args:
        num_points: trajectory length
        kwargs:     any function parameters, not used here

    Returns:
        1d array of generated values; here: randoms in [0,1]
    """
    return np.random.random(num_points)


def base_generator_parameters_fn(**kwargs):
    """
    Base parameters generating function. Provides arguments for data generating function.
    It itself accept arguments specified via `generator_parameters_config` dictionart;

    Returns:
        dictionary of kwargs consistent with generating function used.
    """
    return dict()


class BaseDataGenerator:
    """
    Base synthetic data provider class.
    """
    def __init__(
            self,
            episode_duration=None,
            timeframe=1,
            generator_fn=base_generator_fn,
            generator_parameters_fn=base_generator_parameters_fn,
            generator_parameters_config=None,
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
            episode_duration:               dict, duration of episode in days/hours/mins
            generator_fn                    callabale, should return generated data as 1D np.array
            generator_parameters_fn:        callable, should return dictionary of generator_fn kwargs
            generator_parameters_config:    dict, generator_parameters_fn args
            timeframe:                      int, data periodicity in minutes
            name:                           str
            data_names:                     iterable of str
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
        self.generator_parameters_fn = generator_parameters_fn

        if generator_parameters_config is not None:
            self.generator_parameters_config = generator_parameters_config

        else:
            self.generator_parameters_config = {}

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
        # We need positive datapoints only due to backtrader limitation:
        negs = data_array[data_array < 0]
        if negs.any():
            self.log.warning('{} negative generated values has been set to zero'.format(negs.shape[0]))
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
            Dataset instance with number of records ~ max_episode_len.

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


class TruncatedTestTrial(BTgymDataTrial):
    """
    Utility Trial class without test period: always samples from train,
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
        super(TruncatedTestTrial, self).__init__(
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


class BaseCombinedDataGenerator(BaseDataGenerator):
    """
    Data provider class with synthetic train and real test data.
    """
    def __init__(
            self,
            filename=None,
            parsing_params=None,
            episode_duration_train=None,
            episode_duration_test=None,
            time_gap=None,
            start_00=False,
            name='CombinedDataGenerator',
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
            **kwargs:

        """
        super(BaseCombinedDataGenerator, self).__init__(episode_duration=episode_duration_train, name=name, **kwargs)
        self.nested_params_test = dict(
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=dict(
                sample_duration=episode_duration_test,
                time_gap=time_gap,
                start_00=start_00,
                test_period={'days': 0, 'hours': 0, 'minutes': 0},
            ),
        )
        self.nested_params_test.update(self.nested_params)
        self.nested_class_ref_test = TruncatedTestTrial

    def __read_csv(self, data_filename=None, force_reload=False):
        """
        Populates instance by loading data: CSV file --> pandas dataframe.

        Args:
            data_filename: [opt] csv data filename as string or list of such strings.
            force_reload:  ignore loaded data.
        """
        if self.filename is None and data_filename is None:
            self.data = self.generate_data(self.generator_parameters_fn(**self.generator_parameters_config))
            return

        if self.data is not None and not force_reload:
            data_range = pd.to_datetime(self.data.index)
            self.total_num_records = self.data.shape[0]
            self.data_range_delta = (data_range[-1] - data_range[0]).to_pytimedelta()
            self.log.debug('data has been already loaded. Use `force_reload=True` to reload')
            return

        if data_filename:
            self.filename = data_filename  # override data source if one is given
        if type(self.filename) == str:
            self.filename = [self.filename]

        dataframes = []
        for filename in self.filename:
            try:
                assert filename and os.path.isfile(filename)
                current_dataframe = pd.read_csv(
                    filename,
                    sep=self.sep,
                    header=self.header,
                    index_col=self.index_col,
                    parse_dates=self.parse_dates,
                    names=self.names
                )

                # Check and remove duplicate datetime indexes:
                duplicates = current_dataframe.index.duplicated(keep='first')
                how_bad = duplicates.sum()
                if how_bad > 0:
                    current_dataframe = current_dataframe[~duplicates]
                    self.log.warning('Found {} duplicated date_time records in <{}>.\
                      Removed all but first occurrences.'.format(how_bad, filename))

                dataframes += [current_dataframe]
                self.log.info('Loaded {} records from <{}>.'.format(dataframes[-1].shape[0], filename))

            except:
                msg = 'Data file <{}> not specified / not found.'.format(str(filename))
                self.log.error(msg)
                raise FileNotFoundError(msg)

        self.data = pd.concat(dataframes)
        data_range = pd.to_datetime(self.data.index)
        self.total_num_records = self.data.shape[0]
        self.data_range_delta = (data_range[-1] - data_range[0]).to_pytimedelta()

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
                    'Attempted to sample type {} given current sample type {}, overridden.'.format(
                        self.metadata['type'],
                        sample_type
                    )
                )
                sample_type = self.metadata['type']
        if sample_type:
            # Got test, need natural-born data:
            self.sample_instance = self.nested_class_ref_test(**self.nested_params_test)
            assert self.sample_instance.filename is not None,\
                'Can`t get test sample: test data filename has not been provided.'

            self.log.info('New test sample id: <{}>.'.format(self.sample_instance.filename))
            self.sample_instance.metadata['generator'] = {}

        else:
            self.sample_instance = self.sample_synthetic(sample_type)

        # Common metadata:
        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = self.metadata['sample_num']
        self.sample_instance.metadata['parent_sample_type'] = self.metadata['type']
        self.sample_num += 1

        return self.sample_instance


class BaseTwoLinesCombinedDataGenerator(BTgymMultiData):
    """
    Provides two streams of simulated train spread data, real test data.
    """
    def __init__(
            self,
            assets_filenames=None,
            data_names=None,
            level_process_config=None,
            spread_process_config=None,
            name='2LinesCombinedDataGenerator',
            **kwargs
    ):
        if level_process_config is None:
            self.level_process_config = {
                'generator_fn': base_generator_fn,
                'generator_parameters_fn': base_generator_parameters_fn,
                'generator_parameters_config': None,
            }
        else:
            self.level_process_config = level_process_config

        if spread_process_config is None:
            self.spread_process_config = {
                'generator_fn': base_generator_fn,
                'generator_parameters_fn': base_generator_parameters_fn,
                'generator_parameters_config': None,
            }
        else:
            self.spread_process_config = spread_process_config

        if assets_filenames is None:
            try:
                assert len(data_names) == 2
            except (AssertionError, TypeError) as e:
                raise ValueError('two `data_names` should be provided when `assets_filenames` not specified')
            assets = data_names
            data_config = {asset: {'filename': None} for asset in assets}

        else:
            assert isinstance(assets_filenames, dict) and len(assets_filenames.keys()) == 2, \
                'Expected `assets_filenames` be dict. containing 2 assets names as keys and filenames as values'

            assets = list(assets_filenames.keys())
            data_config = {asset: {'filename': filename} for asset, filename in assets_filenames.items()}

        self.level_asset = assets[0]
        self.spread_asset = assets[-1]

        # Let first asset hold level generating process:
        data_config[self.level_asset].update(self.level_process_config)
        # Second asset will hold spread generating process:
        data_config[self.spread_asset].update(self.spread_process_config)

        super(BaseTwoLinesCombinedDataGenerator, self).__init__(
            data_class_ref=BaseCombinedDataGenerator,
            data_config=data_config,
            assets=assets,
            name=name,
            **kwargs
        )

    def sample(self, sample_type=0, **kwargs):
        if sample_type:
            self._sample_train(sample_type=sample_type, **kwargs)
        else:
            self._sample_test(sample_type=sample_type, **kwargs)

    def _sample_test(self, **kwargs):

        # Get sample to infer exact interval:
        master_sample = self.master_data.sample(**kwargs)

        self.log.debug('master_sample_data: {}'.format(master_sample.data))

        # Prepare empty instance of multistream data:
        sample = BaseTwoLinesCombinedDataGenerator(
            level_process_config=self.level_process_config,
            spread_process_config=self.spread_process_config,
            data_names=self.data_names,
            task=self.task,
            log_level=self.log_level,
            name='sub_' + self.name,
        )
        sample.metadata = copy.deepcopy(master_sample.metadata)

        self.log.debug('sample.metadata: {}'.format(sample.metadata))
        kwargs['interval'] = [sample.metadata['first_row'], sample.metadata['last_row']]

        # Populate sample with data:
        for key, stream in self.data.items():
            sample.data[key] = stream.sample(force_interval=True, **kwargs)

        self.filename = {key: stream.filename for key, stream in self.data.items()}

        return sample

    def _sample_train(self, **kwargs):
        # Get level trajectory:
        level_sample = self.data[self.level_asset].sample(**kwargs)

        # Get spread trajectory:
        spread_sample = self.data[self.spread_asset].sample(**kwargs)

        # Combine and populate

        pass

