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
import os

import backtrader.feeds as btfeeds
import pandas as pd


class BTgymDataset:
    """
    Backtrader.feeds class wrapper.
    Currently pipes CSV[source]-->pandas[for efficient sampling]-->bt.feeds routine.
    Implements random episode data sampling.
    Suggested usage:
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
    # TODO: implement sequential and time-window sampling.
    #  Parameters and their default values:
    params = dict(
        filename=None,  # Should be given either here  or when calling read_csv()

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
        episode_len_days=1,  # Maximum episode time duration in days, hours, minutes:
        episode_len_hours=23,
        episode_len_minutes=55,
        time_gap_days=0,  # Maximum data time gap allowed within sample in days, hours. Thereby,
        time_gap_hours=5,  # if set to be < 1 day, samples containing weekends and holidays gaps will be rejected.
    )
    # Other:
    log = None
    data = None  # Will hold actual data as pandas dataframe
    data_stat = None  # Dataset descriptive statistic as pandas dataframe

    def __init__(self, **kwargs):
        """ _____"""
        # Update parameters with relevant kwargs:
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key] = value

        # Unpack it as attributes:
        for key, value in self.params.items():
                setattr(self, key, value)

        # Maximum data time gap allowed within sample as pydatetimedelta obj:
        self.max_time_gap = datetime.timedelta(days=self.time_gap_days,
                                               hours=self.time_gap_hours,
                                               minutes=0, )
        # ... maximum episode time duration:
        self.max_episode_len = datetime.timedelta(days=self.episode_len_days,
                                                  hours=self.episode_len_hours,
                                                  minutes=self.episode_len_minutes)

        # To log or not to log:
        if not self.log:
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

    def read_csv(self, filename=None):
        """
        Populates instance by loading data: CSV file --> pandas dataframe
        """
        if filename:
            self.filename = filename  # override data source if one is given
        try:
            assert self.filename and os.path.isfile(self.filename)
            self.data = pd.read_csv(self.filename,
                                    sep=self.sep,
                                    header=self.header,
                                    index_col=self.index_col,
                                    parse_dates=self.parse_dates,
                                    names=self.names)
            self.log.info('Loaded {} records from <{}>.'.format(self.data.shape[0], self.filename))

        except:
            try:
                assert 'episode_dataset' in self.filename
                self.log.warning('Attempt to load data into episode dataset: ignored.')
                return None

            except:
                msg = 'Data file <{}> not specified / not found.'.format(str(self.filename))
                self.log.error(msg)
                raise FileNotFoundError(msg)

    def describe(self):
        """
        Returns summary dataset statistic as pandas dataframe:
            records count,
            data mean,
            data std dev,
            min value,
            25% percentile,
            50% percentile,
            75% percentile,
            max value
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
        Returns bt.datafeed instance.
        """
        try:
            assert not self.data.empty
            btfeed = btfeeds.PandasDirectData(dataname=self.data,
                                              timeframe=self.timeframe,
                                              datetime=self.datetime,
                                              open=self.open,
                                              high=self.high,
                                              low=self.low,
                                              close=self.close,
                                              volume=self.volume,
                                              openinterest=self.openinterest,)
            btfeed.numrecords = self.data.shape[0]
            return btfeed

        except:
            msg = 'BTgymDataset instance holds no data. Hint: forgot to call .read_csv()?'
            self.log.info(msg)
            raise AssertionError(msg)

    def sample_random(self):
        """
        Randomly samples continuous subset of data and
        returns BTgymDataset() instance, containing continuous data episode with
        number of records ~ max_episode_len.
        """
        # Maximum possible number of data records (rows) within episode:
        self.episode_num_records = int(self.max_episode_len.total_seconds() / (60 * self.timeframe))

        self.log.info('Maximum episode time duration set to: {}.'.format(self.max_episode_len))
        self.log.info('Respective number of steps: {}.'.format(self.episode_num_records))
        self.log.info('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:

            # Randomly sample record (row) from entire datafeed:
            first_row = int((self.data.shape[0] - self.episode_num_records - 1) * random.random())
            episode_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.info('Episode start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))

            # Keep sampling until good day:
            while not episode_first_day.weekday() in self.start_weekdays:
                self.log.info('Not a good day to start, resampling...')
                first_row = int((self.data.shape[0] - self.episode_num_records - 1) * random.random())
                episode_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.info('Episode start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))
                attempts +=1

            # If 00 option set, get index of first record of that day:
            if self.start_00:
                adj_timedate = episode_first_day.date()
                self.log.info('Start time adjusted to <00:00>')

            else:
                adj_timedate = episode_first_day

            first_row = self.data.index.get_loc(adj_timedate, method='nearest')

            # Easy part:
            last_row = first_row + self.episode_num_records  # + 1
            episode_sample = self.data[first_row: last_row]
            episode_sample_len = (episode_sample.index[-1] - episode_sample.index[0]).to_pytimedelta()
            self.log.info('Episode duration: {}.'.format(episode_sample_len, ))
            self.log.info('Total episode time gap: {}.'.format(episode_sample_len - self.max_episode_len))

            # Perform data gap check:
            if episode_sample_len - self.max_episode_len < self.max_time_gap:
                self.log.info('Sample accepted.')
                # If sample OK - return episodic-dataset:
                episode = BTgymDataset(**self.params)
                episode.filename = '_episode_dataset_' + str(adj_timedate)
                self.log.debug('Episode <filename>: {}'.format(episode.filename))
                episode.data = episode_sample
                return episode

            else:
                self.log.info('Duration too big, resampling...\n')
                attempts += 1

        # Got here -> sanity check failed:
        msg = ('Quitting after {} sampling attempts.' +
               'Hint: check sampling params / dataset consistency.').format(attempts)
        self.log.info(msg)
        raise RuntimeError(msg)
