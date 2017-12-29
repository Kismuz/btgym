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

import logging
#logging.basicConfig(format='%(name)s: %(message)s')

import datetime
import random
from numpy.random import beta as random_beta
import math
import os

import backtrader.feeds as btfeeds
import pandas as pd


class BTgymDataset:
    """
    Base Backtrader.feeds data provider class.

    Enables Pipe::

     CSV[source]-->pandas[for efficient sampling]-->bt.feeds routine.

    Implements random and positional episode data sampling.

    Suggested usage::

        ---user defined ---
        Dataset = BTgymDataset(<filename>,<params>)
        ---inner BTgymServer routine---
        Dataset.read_csv(<filename>)
        Repeat until bored:
            EpisodeDataset = Dataset.get_sample()
            DataFeed = EpisodeDataset.to_btfeed()
            Engine = bt.Cerebro()
            Engine.adddata(DataFeed)
            Engine.run()
    """
    #  Parameters and their default values:
    params = dict(
        filename=None,  # Str or list of str, should be given either here  or when calling read_csv()

        # Default parameters for source-specific CSV datafeed class,
        # correctly parses 1 minute Forex generic ASCII
        # data files from www.HistData.com:

        # CSV to Pandas params.
        sep=';',
        header=0,
        index_col=0,
        parse_dates=True,
        names=['open', 'high', 'low', 'close', 'volume'],

        # Pandas to BT.feeds params:
        timeframe=1,  # 1 minute.
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=-1,
        openinterest=-1,

        # Random-sampling params:
        start_weekdays=[0, 1, 2, 3, ],  # Only weekdays from the list will be used for episode start.
        start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).
        episode_duration=dict(  # Maximum episode time duration in days, hours, minutes:
            days=1,
            hours=23,
            minutes=55
        ),
        time_gap=dict(  # Maximum data time gap allowed within sample in days, hours. Thereby,
            days=0,     # if set to be < 1 day, samples containing weekends and holidays gaps will be rejected.
            hours=5,
        )
    )
    params_deprecated=dict(
        # Deprecated:
        episode_len_days=('episode_duration', 'days'),
        episode_len_hours=('episode_duration','hours'),
        episode_len_minutes=('episode_duration', 'minutes'),
        time_gap_days=('time_gap', 'days'),
        time_gap_hours=('time_gap', 'hours')
    )
    # Other:
    log = None
    data = None  # Will hold actual data as pandas dataframe
    is_ready = False
    data_stat = None  # Dataset descriptive statistic as pandas dataframe
    data_range_delta = None  # Dataset total duration timedelta
    episode_num_records = 0
    metadata = {}

    def __init__(self, **kwargs):
        """

        Args:

            filename:                       Str or list of str, should be given either here or when calling read_csv(),
                                            see `Notes`.

            specific_params CSV to Pandas parsing

            sep:                            ';'
            header:                         0
            index_col:                      0
            parse_dates:                    True
            names:                          ['open', 'high', 'low', 'close', 'volume']

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

            start_weekdays:                 [0, 1, 2, 3, ] - Only weekdays from the list will be used for episode start.
            start_00:                       True - Episode start time will be set to first record of the day
                                            (usually 00:00).
            episode_duration:               {'days': 1, 'hours': 23, 'minutes': 55} - Maximum episode time duration
                                            in days, hours, minutes

            time_gap:                       {'hours': 5} - Data omittance threshold: maximum data time gap allowed
                                            within sample in days, hours. Thereby, if set to be < 1 day,
                                            samples containing weekends and holidays gaps will be rejected.

        Note:
            - CSV file can contain duplicate records, cheks will be performed and all duplicates will be removed;

            - CSV file should be properly sorted by date_time in ascending order, no sorting checks performed.

            - When supplying list of file_names, all files should be also listed ascending by their time period,
              no correct sampling will be possible otherwise.

            - Default parameters are source-specific and made to correctly parse 1 minute Forex generic ASCII
              data files from www.HistData.com. Tune according to your data source.
        """
        # To log or not to log:
        try:
            self.log = kwargs.pop('log')

        except KeyError:
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

        # Update parameters with relevant kwargs:
        self.update_params(**kwargs)

    def update_params(self, **kwargs):
        """
        Updates instance parameters.

        Args:
            **kwargs:   any self.params entries
        """
        self.is_ready = False

        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key] = value

            elif key in self.params_deprecated.keys():
                self.log.warning(
                    'Key: <{}> is deprecated, use: <{}> instead'.
                        format(key, self.params_deprecated[key])
                )
                key1, key2 = self.params_deprecated[key]
                self.params[key1][key2] = value

        # Unpack it as attributes:
        for key, value in self.params.items():
            setattr(self, key, value)

        # Maximum data time gap allowed within sample as pydatetimedelta obj:
        self.max_time_gap = datetime.timedelta(**self.time_gap)

        # ... maximum episode time duration:
        self.max_episode_len = datetime.timedelta(**self.episode_duration)

        # Maximum possible number of data records (rows) within episode:
        self.episode_num_records = int(self.max_episode_len.total_seconds() / (60 * self.timeframe))

    def reset(self, data_filename=None, **kwargs):
        """
        Gets instance ready.

        Args:
            data_filename:  [opt] string or list of strings.
            kwargs:         not used.

        Returns:

        """
        self.read_csv(data_filename)
        self.is_ready = True

    def read_csv(self, data_filename=None):
        """
        Populates instance by loading data: CSV file --> pandas dataframe.

        Args:
            data_filename: [opt] csv data filename as string or list of such strings.
        """
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
                try:
                    assert 'episode_dataset' in self.filename
                    self.log.warning('Attempt to load data into episode dataset: ignored.')
                    return None

                except:
                    msg = 'Data file <{}> not specified / not found.'.format(str(filename))
                    self.log.error(msg)
                    raise FileNotFoundError(msg)

        self.data = pd.concat(dataframes)
        range = pd.to_datetime(self.data.index)
        self.data_range_delta = (range[-1] - range[0]).to_pytimedelta()

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

        except:
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
        Performs BTgymDataset-->bt.feed conversion.

        Returns:
             bt.datafeed instance.
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
            return btfeed

        except (AssertionError, AttributeError) as e:
            msg = 'BTgymDataset instance holds no data. Hint: forgot to call .read_csv()?'
            self.log.error(msg)
            raise AssertionError(msg)

    def sample(self, **kwargs):
        """
        Randomly samples continuous subset of data.

        Args:
            **kwargs:   not used.

        Returns:
             BTgymDataset instance with number of records ~ max_episode_len,
             where `~` tolerance is set by `time_gap` param.
        """
        try:
            assert self.is_ready

        except AssertionError:
            return 'Data not ready. Call .reset() first.'

        episode = self._sample_random()
        episode.metadata['type'] = False  # always `train`
        episode.metadata['trial_num'] = False
        episode.metadata['sample_num'] = False
        return episode

    def _sample_random(self):
        """
        Randomly samples continuous subset of data.

        Returns:
             BTgymDataset instance with number of records ~ max_episode_len,
             where `~` tolerance is set by `time_gap` param.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            raise  AssertionError('BTgymDataset instance holds no data. Hint: forgot to call .read_csv()?')

        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_episode_len))
        self.log.debug('Respective number of steps: {}.'.format(self.episode_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:

            # Randomly sample record (row) from entire datafeed:
            first_row = int((self.data.shape[0] - self.episode_num_records - 1) * random.random())
            episode_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug('Episode start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))

            # Keep sampling until good day:
            while not episode_first_day.weekday() in self.start_weekdays and attempts <= max_attempts:
                self.log.debug('Not a good day to start, resampling...')
                first_row = int((self.data.shape[0] - self.episode_num_records - 1) * random.random())
                episode_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug('Episode start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))
                attempts +=1

            # Check if managed to get proper weekday:
            assert attempts <= max_attempts, \
                'Quitting after {} sampling attempts. Hint: check sampling params / dataset consistency.'. \
                format(attempts)

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = episode_first_day.date()
                self.log.debug('Start time adjusted to <00:00>')

            else:
                adj_timedate = episode_first_day

            first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + self.episode_num_records  # + 1
            episode_sample = self.data[first_row: last_row]
            episode_sample_len = (episode_sample.index[-1] - episode_sample.index[0]).to_pytimedelta()
            self.log.debug('Episode duration: {}.'.format(episode_sample_len, ))
            self.log.debug('Total episode time gap: {}.'.format(episode_sample_len - self.max_episode_len))

            # Perform data gap check:
            if episode_sample_len - self.max_episode_len < self.max_time_gap:
                self.log.debug('Sample accepted.')
                # If sample OK - return episodic-dataset:
                episode = self.__class__(**self.params)
                episode.filename = '_btgym_random_sample_' + str(adj_timedate)
                self.log.info('Episode id: <{}>.'.format(episode.filename))
                episode.data = episode_sample
                episode.metadata['type'] = 'random_sample'
                episode.metadata['first_row'] = first_row
                return episode

            else:
                self.log.debug('Duration too big, resampling...\n')
                attempts += 1

        # Got here -> sanity check failed:
        msg = ('Quitting after {} sampling attempts.' +
               'Hint: check sampling params / dataset consistency.').format(attempts)
        self.log.error(msg)
        raise RuntimeError(msg)

    def _sample_interval(self, interval, b_alpha=1, b_beta=1):
        """
        Samples continuous subset of data,
        such as entire episode records lie within positions specified by interval or.
        Episode start position within interval is drawn from beta-distribution parametrised by `b_alpha, b_beta`.
        By default distribution is uniform one.

        Args:
            interval:       tuple, list or 1d-array of integers of length 2: [lower_position, upper_position];
            b_alpha:        sampling B-distribution alpha param;
            b_beta:         sampling B-distribution beta param;


        Returns:
             - BTgymDataset instance such as:
                1. number of records ~ max_episode_len, subj. to `time_gap` param;
                2. actual episode start position is sampled from `interval`;
             - `False` if it is not possible to sample instance with set args.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            raise  AssertionError('BTgymDataset instance holds no data. Hint: forgot to call .read_csv()?')

        assert len(interval) == 2, 'Invalid interval arg: expected list or tuple of size 2, got: {}'.format(interval)

        sample_num_records = self.episode_num_records

        assert interval[0] < interval[-1] < int(self.data.shape[0] - sample_num_records), \
            'Cannot sample with size {}, in {} from dataset of {} records'.\
             format(sample_num_records, interval, self.data.shape[0])

        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_episode_len))
        self.log.debug('Respective number of steps: {}.'.format(sample_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:

            first_row = interval[0] + round(
                (interval[-1] - interval[0] - sample_num_records - 1) * random_beta(a=b_alpha, b=b_beta)
            )

            episode_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug('Sample start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))

            # Keep sampling until good day:
            while not episode_first_day.weekday() in self.start_weekdays and attempts <= max_attempts:
                self.log.debug('Not a good day to start, resampling...')
                first_row = interval[0] + round(
                    (interval[-1] - interval[0] - sample_num_records - 1) * random_beta(a=b_alpha, b=b_beta)
                )
                episode_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug('Sample start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))
                attempts += 1

            # Check if managed to get proper weekday:
            assert attempts <= max_attempts, \
                'Quitting after {} sampling attempts. Hint: check sampling params / dataset consistency.'.\
                format(attempts)

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = episode_first_day.date()
                self.log.debug('Start time adjusted to <00:00>')

            else:
                adj_timedate = episode_first_day

            first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + sample_num_records  # + 1
            episode_sample = self.data[first_row: last_row]
            episode_sample_len = (episode_sample.index[-1] - episode_sample.index[0]).to_pytimedelta()
            self.log.debug('Sample duration: {}.'.format(episode_sample_len, ))
            self.log.debug('Total sample time gap: {}.'.format(episode_sample_len - self.max_episode_len))

            # Perform data gap check:
            if episode_sample_len - self.max_episode_len < self.max_time_gap:
                self.log.debug('Sample accepted.')
                # If sample OK - return episodic-dataset:
                episode = self.__class__(**self.params)
                episode.filename = '_btgym_interval_sample_' + str(adj_timedate)
                self.log.info('Sample id: <{}>.'.format(episode.filename))
                episode.data = episode_sample
                episode.metadata['type'] = 'interval_sample'
                episode.metadata['first_row'] = first_row
                return episode

            else:
                self.log.debug('Attempt {}: duration too big, resampling, ...\n'.format(attempts))
                attempts += 1

        # Got here -> sanity check failed:
        msg = ('Quitting after {} sampling attempts.' +
               'Hint: check sampling params / dataset consistency.').format(attempts)
        self.log.warning(msg)
        raise AssertionError(msg)

