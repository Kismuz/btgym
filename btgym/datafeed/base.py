###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
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

import copy
import datetime
import random

import backtrader.feeds as btfeeds
import pandas as pd
import sys
from backtrader import TimeFrame
from logbook import Logger, StreamHandler, WARNING
from numpy.random import beta as random_beta

DataSampleConfig = dict(
    get_new=True,
    sample_type=0,
    timestamp=None,
    b_alpha=1,
    b_beta=1
)
"""
dict: Conventional sampling configuration template to pass to data class `sample()` method:

```sample = my_data.sample(**DataSampleConfig)```
"""


EnvResetConfig = dict(
    episode_config=copy.deepcopy(DataSampleConfig),
    trial_config=copy.deepcopy(DataSampleConfig),
)
"""
dict: Conventional reset configuration template to pass to environment `reset()` method:

```observation = env.reset(**EnvResetConfig)```
"""


class BTgymBaseData:
    """
    Base BTgym data provider class.
    Provides core sampling, splitting  and converting functionality.
    Do not use directly.

    Enables Pipe::

        pandas[for efficient sampling]-->bt.feeds

    """

    def __init__(
            self,
            dataframe=None,
            parsing_params=None,
            sampling_params=None,
            name='base_data',
            data_names=('default_asset',),
            task=0,
            frozen_time_split=None,
            log_level=WARNING,
            _config_stack=None,
            **kwargs
    ):
        """
        Args:

            dataframe:                      pd.dataframe holding data

            specific_params Pandas to BT.feeds conversion

            timeframe=1:                    1 minute.
            datetime:                       0
            open:                           1
            high:                           2
            low:                            3
            close:                          4
            volume:                         -1
            openinterest:                   -1

            specific_params Sampling

            sample_class_ref:               None - if not None, than sample() method will return instance of specified
                                            class, which itself must be subclass of BaseBTgymDataset,
                                            else returns instance of the base data class.

            start_weekdays:                 [0, 1, 2, 3, ] - Only weekdays from the list will be used for sample start.
            start_00:                       True - sample start time will be set to first record of the day
                                            (usually 00:00).
            sample_duration:                {'days': 1, 'hours': 23, 'minutes': 55} - Maximum sample time duration
                                            in days, hours, minutes
            time_gap:                       {''days': 0, hours': 5, 'minutes': 0} - Data omittance threshold:
                                            maximum no-data time gap allowed within sample in days, hours.
                                            Thereby, if set to be < 1 day, samples containing weekends and holidays gaps
                                            will be rejected.
            test_period:                    {'days': 0, 'hours': 0, 'minutes': 0} - setting this param to non-zero
                                            duration forces instance.data split to train / test subsets with test
                                            subset duration equal to `test_period` with `time_gap` tolerance. Train data
                                            always precedes test one:
                                            [0_record<-train_data->split_point_record<-test_data->last_record].
            sample_expanding:               None, reserved for child classes.

        Note:
            - CSV file can contain duplicate records, checks will be performed and all duplicates will be removed;

            - CSV file should be properly sorted by date_time in ascending order, no sorting checks performed.

            - When supplying list of file_names, all files should be also listed ascending by their time period,
              no correct sampling will be possible otherwise.

            - Default parameters are source-specific and made to correctly parse 1 minute Forex generic ASCII
              data files from www.HistData.com. Tune according to your data source.
        """

        self._set_dataframe(dataframe)

        if parsing_params is None:
            self.parsing_params = dict(
                # Pandas to BT.feeds params:
                timeframe=1,  # 1 minute.
                datetime=0,
                open=1,
                high=2,
                low=3,
                close=4,
                volume=-1,
                openinterest=-1,
            )
        else:
            self.parsing_params = parsing_params

        if sampling_params is None:
            self.sampling_params = dict(
                # Sampling params:
                start_weekdays=[],  # Only weekdays from the list will be used for episode start.
                start_00=False,  # Sample start time will be set to first record of the day (usually 00:00).
                sample_duration=dict(  # Maximum sample time duration in days, hours, minutes:
                    days=0,
                    hours=0,
                    minutes=0
                ),
                time_gap=dict(  # Maximum data time gap allowed within sample in days, hours. Thereby,
                    days=0,  # if set to be < 1 day, samples containing weekends and holidays gaps will be rejected.
                    hours=0,
                ),
                test_period=dict(  # Time period to take test samples from, in days, hours, minutes:
                    days=0,
                    hours=0,
                    minutes=0
                ),
                expanding=False,
            )
        else:
            self.sampling_params = sampling_params

        self.name = name
        # String will be used as key name for bt_feed data-line:

        self.task = task
        self.log_level = log_level
        self.data_names = data_names
        self.data_name = self.data_names[0]

        self.is_ready = False

        self.global_timestamp = 0
        self.start_timestamp = 0
        self.final_timestamp = 0

        self.data_stat = None  # Dataset descriptive statistic as pandas dataframe
        self.max_time_gap = None
        self.time_gap = None
        self.max_sample_len_delta = None
        self.sample_duration = None
        self.sample_num_records = 0
        self.start_weekdays = {0, 1, 2, 3, 4, 5, 6}
        self.start_00 = False
        self.expanding = False

        self.sample_instance = None

        self.test_range_delta = None
        self.train_range_delta = None
        self.test_num_records = 0
        self.train_num_records = 0
        self.train_interval = [0, 0]
        self.test_interval = [0, 0]
        self.test_period = {'days': 0, 'hours': 0, 'minutes': 0}
        self.train_period = {'days': 0, 'hours': 0, 'minutes': 0}
        self._test_period_backshift_delta = datetime.timedelta(**{'days': 0, 'hours': 0, 'minutes': 0})
        self.sample_num = 0
        self.task = 0
        self.metadata = {'sample_num': 0, 'type': None}

        self.set_params(self.parsing_params)
        self.set_params(self.sampling_params)

        self._config_stack = copy.deepcopy(_config_stack)
        try:
            nested_config = self._config_stack.pop()

        except (IndexError, AttributeError) as e:
            # IF stack is empty, sample of this instance itself is not supposed to be sampled.
            nested_config = dict(
                class_ref=None,
                kwargs=dict(
                    parsing_params=self.parsing_params,
                    sample_params=None,
                    name='data_stream',
                    task=self.task,
                    log_level=self.log_level,
                    _config_stack=None,
                )
            )
        # Configure sample instance parameters:
        self.nested_class_ref = nested_config['class_ref']
        self.nested_params = nested_config['kwargs']
        self.sample_name = '{}_w_{}_'.format(self.nested_params['name'], self.task)
        self.nested_params['_config_stack'] = self._config_stack

        # Logging:
        StreamHandler(sys.stdout).push_application()
        self.set_logger(self.log_level, self.task)

        # Legacy parameter dictionary, left here for BTgym API_shell:
        self.params = {}
        self.params.update(self.parsing_params)
        self.params.update(self.sampling_params)

        if frozen_time_split is not None:
            self.frozen_time_split = datetime.datetime(**frozen_time_split)

        else:
            self.frozen_time_split = None

        self.frozen_split_timestamp = None

    def _set_dataframe(self, dataframe: pd.DataFrame):
        if dataframe is not None:
            self.data = dataframe
        else:
            raise AssertionError("Data frame has to be defined.")
        if self.data.empty:
            raise AssertionError("DataFrame holds no data.")

    def set_params(self, params_dict):
        """
        Batch attribute setter.

        Args:
            params_dict: dictionary of parameters to be set as instance attributes.
        """
        for key, value in params_dict.items():
            setattr(self, key, value)

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

    def set_global_timestamp(self, timestamp):
        if self.data is not None:
            self.global_timestamp = self.data.index[0].timestamp()

    def reset(self, **kwargs):
        """
        Gets instance ready.

        Args:
            kwargs:         not used.

        """
        self._reset(**kwargs)

    def _reset(self, timestamp=None, **kwargs):

        # Add global timepoints:
        self.start_timestamp = self.data.index[0].timestamp()
        self.final_timestamp = self.data.index[-1].timestamp()

        if self.frozen_time_split is not None:
            frozen_index = self.data.index.get_loc(self.frozen_time_split, method='ffill')
            self.frozen_split_timestamp = self.data.index[frozen_index].timestamp()
            self.set_global_timestamp(self.frozen_split_timestamp)

        else:
            self.frozen_split_timestamp = None
            self.set_global_timestamp(timestamp)

        self.log.debug(
            'time stamps start: {}, current: {} final: {}'.format(
                self.start_timestamp,
                self.global_timestamp,
                self.final_timestamp
            )
        )

        # Maximum data time gap allowed within sample as pydatetimedelta obj:
        self.max_time_gap = datetime.timedelta(**self.time_gap)

        # Max. gap number of records:
        self.max_gap_num_records = int(self.max_time_gap.total_seconds() / (60 * self.timeframe))

        # ... maximum episode time duration:
        self.max_sample_len_delta = datetime.timedelta(**self.sample_duration)

        # Maximum possible number of data records (rows) within episode:
        self.sample_num_records = int(self.max_sample_len_delta.total_seconds() / (60 * self.timeframe))

        self.backshift_num_records = round(self._test_period_backshift_delta.total_seconds() / (60 * self.timeframe))

        # Train/test timedeltas:
        if self.train_period is None or self.test_period == -1:
            # No train data assumed, test only:
            self.train_num_records = 0
            self.test_num_records = self.data.shape[0] - self.backshift_num_records
            break_point = self.backshift_num_records
            self.train_interval = [0, 0]
            self.test_interval = [self.backshift_num_records, self.data.shape[0]]

        else:
            # Train and maybe test data assumed:
            if self.test_period is not None:
                self.test_range_delta = datetime.timedelta(**self.test_period)
                self.test_num_records = round(self.test_range_delta.total_seconds() / (60 * self.timeframe))
                self.train_num_records = self.data.shape[0] - self.test_num_records
                break_point = self.train_num_records
                self.train_interval = [0, break_point]
                self.test_interval = [break_point - self.backshift_num_records, self.data.shape[0]]
            else:
                self.test_num_records = 0
                self.train_num_records = self.data.shape[0]
                break_point = self.train_num_records
                self.train_interval = [0, break_point]
                self.test_interval = [0, 0]

        if self.train_num_records > 0:
            try:
                assert self.train_num_records + self.max_gap_num_records >= self.sample_num_records

            except AssertionError:
                self.log.exception(
                    'Train subset should contain at least one sample, ' +
                    'got: train_set size: {} rows, sample_size: {} rows, tolerance: {} rows'.
                    format(self.train_num_records, self.sample_num_records, self.max_gap_num_records)
                )
                raise AssertionError

        if self.test_num_records > 0:
            try:
                assert self.test_num_records + self.max_gap_num_records >= self.sample_num_records

            except AssertionError:
                self.log.exception(
                    'Test subset should contain at least one sample, ' +
                    'got: test_set size: {} rows, sample_size: {} rows, tolerance: {} rows'.
                    format(self.test_num_records, self.sample_num_records, self.max_gap_num_records)
                )
                raise AssertionError

        self.sample_num = 0
        self.is_ready = True


    def describe(self): #TODO: remove. There is one dependency, probably not used.
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
        def bt_timeframe(minutes):
            timeframe = TimeFrame.Minutes
            if minutes / 1440 == 1:
                timeframe = TimeFrame.Days
            return timeframe

        btfeed = btfeeds.PandasDirectData(
            dataname=self.data,
            timeframe=bt_timeframe(self.timeframe),
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


    def sample(self, **kwargs):
        return self._sample(**kwargs)

    def _sample(
            self,
            get_new=True,
            sample_type=0,
            b_alpha=1.0,
            b_beta=1.0,
            force_interval=False,
            interval=None,
            **kwargs
    ):
        """
        Samples continuous subset of data.

        Args:
            get_new (bool):                     sample new (True) or reuse (False) last made sample;
            sample_type (int or bool):          0 (train) or 1 (test) - get sample from train or test data subsets
                                                respectively.
            b_alpha (float):                    beta-distribution sampling alpha > 0, valid for train episodes.
            b_beta (float):                     beta-distribution sampling beta > 0, valid for train episodes.
            force_interval(bool):               use exact sampling interval (should be given)
            interval(iterable of int, len2):    exact interval to sample from when force_interval=True

        Returns:
        if no sample_class_ref param been set:
            BTgymDataset instance with number of records ~ max_episode_len,
            where `~` tolerance is set by `time_gap` param;
        else:
            `sample_class_ref` instance with same as above number of records.

        Note:
                Train sample start position within interval is drawn from beta-distribution
                with default parameters b_alpha=1, b_beta=1, i.e. uniform one.
                Beta-distribution makes skewed sampling possible , e.g.
                to give recent episodes higher probability of being sampled, e.g.:  b_alpha=10, b_beta=0.8.
                Test samples are always uniform one.

        """
        try:
            assert self.is_ready

        except AssertionError:
            msg = 'sampling attempt: data not ready. Hint: forgot to call data.reset()?'
            self.log.error(msg)
            raise RuntimeError(msg)

        try:
            assert sample_type in [0, 1]

        except AssertionError:
            msg = 'sampling attempt: expected sample type be in {}, got: {}'.format([0, 1], sample_type)
            self.log.error(msg)
            raise ValueError(msg)

        if force_interval:
            try:
                assert interval is not None and len(list(interval)) == 2

            except AssertionError:
                msg = 'sampling attempt: got force_interval=True, expected interval=[a,b], got: <{}>'.format(interval)
                self.log.error(msg)
                raise ValueError(msg)

        if self.sample_instance is None or get_new:
            if sample_type == 0:
                # Get beta_distributed sample in train interval:
                if force_interval:
                    sample_interval = interval
                else:
                    sample_interval = self.train_interval

                self.sample_instance = self._sample_interval(
                    sample_interval,
                    force_interval=force_interval,
                    b_alpha=b_alpha,
                    b_beta=b_beta,
                    name='train_' + self.sample_name,
                    **kwargs
                )

            else:
                # Get uniform sample in test interval:
                if force_interval:
                    sample_interval = interval
                else:
                    sample_interval = self.test_interval

                self.sample_instance = self._sample_interval(
                    sample_interval,
                    force_interval=force_interval,
                    b_alpha=1,
                    b_beta=1,
                    name='test_' + self.sample_name,
                    **kwargs
                )
            self.sample_instance.metadata['type'] = sample_type  # TODO: can move inside sample()
            self.sample_instance.metadata['sample_num'] = self.sample_num
            self.sample_instance.metadata['parent_sample_num'] = copy.deepcopy(self.metadata['sample_num'])
            self.sample_instance.metadata['parent_sample_type'] = copy.deepcopy(self.metadata['type'])
            self.sample_num += 1

        else:
            # Do nothing:
            self.log.debug('Reusing sample, id: {}'.format(self.sample_instance.filename))

        return self.sample_instance

    def _sample_random(
            self,
            sample_type=0,
            timestamp=None,
            name='random_sample_',
            interval=None,
            force_interval=False,
            **kwargs
    ):
        """
        Randomly samples continuous subset of data.

        Args:
            name:        str, sample filename id

        Returns:
             BTgymDataset instance with number of records ~ max_episode_len,
             where `~` tolerance is set by `time_gap` param.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.log.exception('Instance holds no data. Hint: forgot to call .read_csv()?')
            raise AssertionError

        if force_interval:
            raise NotImplementedError('Force_interval for random sampling not implemented.')

        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_sample_len_delta))
        self.log.debug('Respective number of steps: {}.'.format(self.sample_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        sampled_data = None
        sample_len = 0

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:

            # Randomly sample record (row) from entire datafeed:
            first_row = int((self.data.shape[0] - self.sample_num_records - 1) * random.random())
            sample_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug('Sample start: {}, weekday: {}.'.format(sample_first_day, sample_first_day.weekday()))

            # Keep sampling until good day:
            while not sample_first_day.weekday() in self.start_weekdays and attempts <= max_attempts:
                self.log.debug('Not a good day to start, resampling...')
                first_row = int((self.data.shape[0] - self.sample_num_records - 1) * random.random())
                sample_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug('Sample start: {}, weekday: {}.'.format(sample_first_day, sample_first_day.weekday()))
                attempts +=1

            # Check if managed to get proper weekday:
            assert attempts <= max_attempts, \
                'Quitting after {} sampling attempts. Hint: check sampling params / dataset consistency.'. \
                format(attempts)

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = sample_first_day.date()
                self.log.debug('Start time adjusted to <00:00>')

            else:
                adj_timedate = sample_first_day

            first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + self.sample_num_records  # + 1
            sampled_data = self.data[first_row: last_row]
            sample_len = (sampled_data.index[-1] - sampled_data.index[0]).to_pytimedelta()
            self.log.debug('Actual sample duration: {}.'.format(sample_len, ))
            self.log.debug('Total sample time gap: {}.'.format(self.max_sample_len_delta - sample_len))

            # Perform data gap check:
            if self.max_sample_len_delta - sample_len < self.max_time_gap:
                self.log.debug('Sample accepted.')
                # If sample OK - compose and return sample:
                new_instance = self.nested_class_ref(**self.nested_params)
                new_instance.filename = name + 'n{}_at_{}'.format(self.sample_num, adj_timedate)
                self.log.info('Sample id: <{}>.'.format(new_instance.filename))
                new_instance.data = sampled_data
                new_instance.metadata['type'] = 'random_sample'
                new_instance.metadata['first_row'] = first_row
                new_instance.metadata['last_row'] = last_row

                return new_instance

            else:
                self.log.debug('Duration too big, resampling...\n')
                attempts += 1

        # Got here -> sanity check failed:
        msg = (
            '\nQuitting after {} sampling attempts.\n' +
            'Full sample duration: {}\n' +
            'Total sample time gap: {}\n' +
            'Sample start time: {}\n' +
            'Sample finish time: {}\n' +
            'Hint: check sampling params / dataset consistency.'
        ).format(
            attempts,
            sample_len,
            sample_len - self.max_sample_len_delta,
            sampled_data.index[0],
            sampled_data.index[-1]

        )
        self.log.error(msg)
        raise RuntimeError(msg)

    def _sample_interval(
            self,
            interval,
            b_alpha=1.0,
            b_beta=1.0,
            name='interval_sample_',
            force_interval=False,
            **kwargs
    ):
        """
        Samples continuous subset of data,
        such as entire episode records lie within positions specified by interval.
        Episode start position within interval is drawn from beta-distribution parametrised by `b_alpha, b_beta`.
        By default distribution is uniform one.

        Args:
            interval:       tuple, list or 1d-array of integers of length 2: [lower_row_number, upper_row_number];
            b_alpha:        float > 0, sampling B-distribution alpha param, def=1;
            b_beta:         float > 0, sampling B-distribution beta param, def=1;
            name:           str, sample filename id
            force_interval: bool,  if true: force exact interval sampling


        Returns:
             - BTgymDataset instance such as:
                1. number of records ~ max_episode_len, subj. to `time_gap` param;
                2. actual episode start position is sampled from `interval`;
             - `False` if it is not possible to sample instance with set args.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.log.exception('Instance holds no data. Hint: forgot to call .read_csv()?')
            raise AssertionError

        try:
            assert len(interval) == 2

        except AssertionError:
            self.log.exception(
                'Invalid interval arg: expected list or tuple of size 2, got: {}'.format(interval)
            )
            raise AssertionError

        if force_interval:
            return self._sample_exact_interval(interval, name)

        try:
            assert b_alpha > 0 and b_beta > 0

        except AssertionError:
            self.log.exception(
                'Expected positive B-distribution [alpha, beta] params, got: {}'.format([b_alpha, b_beta])
            )
            raise AssertionError

        if interval[-1] - interval[0] + self.max_gap_num_records > self.sample_num_records:
            sample_num_records = self.sample_num_records
        else:
            sample_num_records = interval[-1] - interval[0]

        self.log.debug('Sample interval: {}'.format(interval))
        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_sample_len_delta))
        self.log.debug('Sample number of steps (adjusted to interval): {}.'.format(sample_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        sampled_data = None
        sample_len = 0

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:

            first_row = interval[0] + int(
                (interval[-1] - interval[0] - sample_num_records) * random_beta(a=b_alpha, b=b_beta)
            )

            #print('_sample_interval_sample_num_records: ', sample_num_records)
            #print('_sample_interval_first_row: ', first_row)

            sample_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug(
                'Sample start row: {}, day: {}, weekday: {}.'.
                format(first_row, sample_first_day, sample_first_day.weekday())
            )

            # Keep sampling until good day:
            while not sample_first_day.weekday() in self.start_weekdays and attempts <= max_attempts:
                self.log.debug('Not a good day to start, resampling...')
                first_row = interval[0] + round(
                    (interval[-1] - interval[0] - sample_num_records) * random_beta(a=b_alpha, b=b_beta)
                )
                #print('r_sample_interval_sample_num_records: ', sample_num_records)
                #print('r_sample_interval_first_row: ', first_row)
                sample_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug(
                    'Sample start row: {}, day: {}, weekday: {}.'.
                    format(first_row, sample_first_day, sample_first_day.weekday())
                )
                attempts += 1

            # Check if managed to get proper weekday:
            try:
                assert attempts <= max_attempts

            except AssertionError:
                self.log.exception(
                    'Quitting after {} sampling attempts. Hint: check sampling params / dataset consistency.'.
                    format(attempts)
                )
                raise RuntimeError

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = sample_first_day.date()
                self.log.debug('Start time adjusted to <00:00>')
                first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            else:
                adj_timedate = sample_first_day

            # first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + sample_num_records  # + 1
            sampled_data = self.data[first_row: last_row]

            self.log.debug(
                'first_row: {}, last_row: {}, data_shape: {}'.format(
                    first_row,
                    last_row,
                    sampled_data.shape
                )
            )
            sample_len = (sampled_data.index[-1] - sampled_data.index[0]).to_pytimedelta()
            self.log.debug('Actual sample duration: {}.'.format(sample_len))
            self.log.debug('Total sample time gap: {}.'.format(self.max_sample_len_delta - sample_len))

            # Perform data gap check:
            if self.max_sample_len_delta - sample_len < self.max_time_gap:
                self.log.debug('Sample accepted.')
                # If sample OK - return new dataset:
                new_instance = self.nested_class_ref(**self.nested_params)
                new_instance.filename = name + 'num_{}_at_{}'.format(self.sample_num, adj_timedate)
                self.log.info('New sample id: <{}>.'.format(new_instance.filename))
                new_instance.data = sampled_data
                new_instance.metadata['type'] = 'interval_sample'
                new_instance.metadata['first_row'] = first_row
                new_instance.metadata['last_row'] = last_row

                return new_instance

            else:
                self.log.debug('Attempt {}: gap is too big, resampling, ...\n'.format(attempts))
                attempts += 1

        # Got here -> sanity check failed:
        msg = (
                '\nQuitting after {} sampling attempts.\n' +
                'Full sample duration: {}\n' +
                'Total sample time gap: {}\n' +
                'Sample start time: {}\n' +
                'Sample finish time: {}\n' +
                'Hint: check sampling params / dataset consistency.'
        ).format(
            attempts,
            sample_len,
            sample_len - self.max_sample_len_delta,
            sampled_data.index[0],
            sampled_data.index[-1]

        )
        self.log.error(msg)
        raise RuntimeError(msg)

    def _sample_aligned_interval(
            self,
            interval,
            align_left=False,
            b_alpha=1.0,
            b_beta=1.0,
            name='interval_sample_',
            force_interval=False,
            **kwargs
    ):
        """
        Samples continuous subset of data,
        such as entire episode records lie within positions specified by interval
        Episode start position within interval is drawn from beta-distribution parametrised by `b_alpha, b_beta`.
        By default distribution is uniform one.

        Args:
            interval:       tuple, list or 1d-array of integers of length 2: [lower_row_number, upper_row_number];
            align:          if True - try to align sample to beginning of interval;
            b_alpha:        float > 0, sampling B-distribution alpha param, def=1;
            b_beta:         float > 0, sampling B-distribution beta param, def=1;
            name:           str, sample filename id
            force_interval: bool,  if true: force exact interval sampling

        Returns:
             - BTgymDataset instance such as:
                1. number of records ~ max_episode_len, subj. to `time_gap` param;
                2. actual episode start position is sampled from `interval`;
             - `False` if it is not possible to sample instance with set args.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.log.exception('Instance holds no data. Hint: forgot to call .read_csv()?')
            raise AssertionError

        try:
            assert len(interval) == 2

        except AssertionError:
            self.log.exception(
                'Invalid interval arg: expected list or tuple of size 2, got: {}'.format(interval)
            )
            raise AssertionError

        if force_interval:
            return self._sample_exact_interval(interval, name)

        try:
            assert b_alpha > 0 and b_beta > 0

        except AssertionError:
            self.log.exception(
                'Expected positive B-distribution [alpha, beta] params, got: {}'.format([b_alpha, b_beta])
            )
            raise AssertionError

        sample_num_records = self.sample_num_records

        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_sample_len_delta))
        self.log.debug('Respective number of steps: {}.'.format(sample_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        # Sanity check param:
        if align_left:
            max_attempts = interval[-1] - interval[0]
        else:
            # Sanity check:
            max_attempts = 100

        attempts = 0
        align_shift = 0

        # Sample enter point as close to beginning  until all conditions are met:
        while attempts <= max_attempts:
            if align_left:
                first_row = interval[0] + align_shift

            else:
                first_row = interval[0] + int(
                    (interval[-1] - interval[0] - sample_num_records) * random_beta(a=b_alpha, b=b_beta)
                )

            #print('_sample_interval_sample_num_records: ', sample_num_records)
            self.log.debug('_sample_interval_first_row: {}'.format(first_row))

            sample_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug('Sample start: {}, weekday: {}.'.format(sample_first_day, sample_first_day.weekday()))

            # Keep sampling until good day:
            while not sample_first_day.weekday() in self.start_weekdays and attempts <= max_attempts:
                align_shift += 1

                self.log.debug('Not a good day to start, resampling...')

                if align_left:
                    first_row = interval[0] + align_shift
                else:

                    first_row = interval[0] + int(
                        (interval[-1] - interval[0] - sample_num_records) * random_beta(a=b_alpha, b=b_beta)
                    )
                #print('r_sample_interval_sample_num_records: ', sample_num_records)
                self.log.debug('_sample_interval_first_row: {}'.format(first_row))

                sample_first_day = self.data[first_row:first_row + 1].index[0]

                self.log.debug('Sample start: {}, weekday: {}.'.format(sample_first_day, sample_first_day.weekday()))

                attempts += 1

            # Check if managed to get proper weekday:
            try:
                assert attempts <= max_attempts

            except AssertionError:
                self.log.exception(
                    'Quitting after {} sampling attempts. Hint: check sampling params / dataset consistency.'.
                    format(attempts)
                )
                raise RuntimeError

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = sample_first_day.date()
                self.log.debug('Start time adjusted to <00:00>')
                first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            else:
                adj_timedate = sample_first_day

            # first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + sample_num_records  # + 1
            sampled_data = self.data[first_row: last_row]
            sample_len = (sampled_data.index[-1] - sampled_data.index[0]).to_pytimedelta()
            self.log.debug('Actual sample duration: {}.'.format(sample_len))
            self.log.debug('Total sample time gap: {}.'.format(sample_len - self.max_sample_len_delta))

            # Perform data gap check:
            if sample_len - self.max_sample_len_delta < self.max_time_gap:
                self.log.debug('Sample accepted.')
                # If sample OK - return new dataset:
                new_instance = self.nested_class_ref(**self.nested_params)
                new_instance.filename = name + 'num_{}_at_{}'.format(self.sample_num, adj_timedate)
                self.log.info('New sample id: <{}>.'.format(new_instance.filename))
                new_instance.data = sampled_data
                new_instance.metadata['type'] = 'interval_sample'
                new_instance.metadata['first_row'] = first_row
                new_instance.metadata['last_row'] = last_row

                return new_instance

            else:
                self.log.debug('Attempt {}: duration too big, resampling, ...\n'.format(attempts))
                attempts += 1
                align_shift += 1

        # Got here -> sanity check failed:
        msg = ('Quitting after {} sampling attempts.' +
               'Hint: check sampling params / dataset consistency.').format(attempts)
        self.log.error(msg)
        raise RuntimeError(msg)

    def _sample_exact_interval(self, interval, name='interval_sample_', **kwargs):
        """
        Samples exactly defined interval.

        Args:
            interval:   tuple, list or 1d-array of integers of length 2: [lower_row_number, upper_row_number];
            name:       str, sample filename id

        Returns:
             BTgymDataset instance.

        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.log.exception('Instance holds no data. Hint: forgot to call .read_csv()?')
            raise AssertionError

        try:
            assert len(interval) == 2

        except AssertionError:
            self.log.exception(
                'Invalid interval arg: expected list or tuple of size 2, got: {}'.format(interval)
            )
            raise AssertionError

        first_row = interval[0]
        last_row = interval[-1]
        sampled_data = self.data[first_row: last_row]

        sample_first_day = self.data[first_row:first_row + 1].index[0]

        new_instance = self.nested_class_ref(**self.nested_params)
        new_instance.filename = name + 'num_{}_at_{}'.format(self.sample_num, sample_first_day)
        self.log.info('New sample id: <{}>.'.format(new_instance.filename))
        new_instance.data = sampled_data
        new_instance.metadata['type'] = 'interval_sample'
        new_instance.metadata['first_row'] = first_row
        new_instance.metadata['last_row'] = last_row

        return new_instance
