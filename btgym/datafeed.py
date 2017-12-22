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
            kwargs:
            filename=None:      Str or list of str, should be given either here or when calling read_csv(), see `Notes`.

            CSV to Pandas parsing params:(pandas specific)

            sep=';'
            header=0,
            index_col=0
            parse_dates=True
            names=['open', 'high', 'low', 'close', 'volume']

            Pandas to BT.feeds params (backtrader specific)

            timeframe=1:                    1 minute.
            datetime=0
            open=1
            high=2
            low=3
            close=4
            volume=-1
            openinterest=-1

            Sampling params

            start_weekdays=[0, 1, 2, 3, ]:  Only weekdays from the list will be used for episode start.
            start_00=True:                  Episode start time will be set to first record of the day (usually 00:00).
            episode_duration={'days': 1, 'hours': 23, 'minutes': 55}:   Maximum episode time duration
                                                                        in days, hours, minutes

            time_gap={'hours': 5}:          Data omittance threshold: maximum data time gap allowed within sample
                                            in days, hours. Thereby, if set to be < 1 day,
                                            samples containing weekends and holidays gaps will be rejected.

        Note:
            - CSV file can contain duplicate records, cheks will be performed and all duplicates will be removed;

            - CSV file should be properly sorted by date_time in ascending order, no sorting checks performed.

            - When supplying list of file_names, all files should be also sorted by their time period,
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


class BTgymSequentialTrial(BTgymDataset):
    """
    Sequential Data Trials iterator.
    Enables sliding or expanding time-window training and testing for the dataset of time-ordered records.

    Single Trial is defined by:

    - support train and test intervals::

        [train_start_time <-> train_end_time], [test_start_time <-> test_end_time],
        such as:
        train_start_time < train_end_time + 1 = test_start_time < test_end_time,
        where `1` stands for next closest time.

    - number of train episodes to draw from train support interval;

    - number of test episodes to draw from test support interval;

    Sliding time-window data iterating:

    If training is started from the beginningg of the dataset, `train_start_time` is set to that of first record,
    for example, for the start of the year::

        Trial train interval: 19 days, 23:59:00; test interval: 2 days, 23:59:00;
        Train episodes per trial: 1000; test episodes per trial: 10, test_period: 50, iterating from 0-th

    Then first trial intervals will be (note that omitted data periods like holidays are excluded)::

        Training interval: 2016-01-03 17:01:00 <--> 2016-01-31 17:14:00;
        Testing  interval: 2016-01-31 17:15:00 <--> 2016-02-03 17:14:00;

    Since `reset_data()` method call, every call to `BTgymSequentialTrial.sample()` method will return randomly drawn
    train episode from train interval, until reached `test_period` number of samples(here -50). Than iterator `pauses
    training' and each next call to `sample()` will return randomly drawn episode from test interval,
    until again max. number is reached (here - 10).
    Train-test loop is repeated until max. number of `Trial` train samples is reached ( here - 1000).

    Next call to `sample()` will result in following: next `Trial` will be formed such as::

        train_start_time_next_trial = `test_end_time_previous_trial + 1

    i.e. `Trial` will be shifted by the duration of test period,
    than first train episode of the new `Trial` will be sampled and returned.

    Repeats until entire dataset is exhausted.

    Note that while train periods are overlapping, test periods form a partition.

    Here, next trial will be::

        Training @: 2016-01-06 00:00:00 <--> 2016-02-03 00:10:00;
        Testing  @: 2016-02-03 00:12:00 <--> 2016-02-08 00:13:00

    Expanding time-window data iterating:

    Differs from above in a way that trial interval start position is fixed at the beginning of dataset. Thus,
    trial support interval is expanding to the right and every subsequent trial is `longer` than previous one
    by amount of test interval.

    Episodes sampling:

    Episodes sampling is performed in such a way that entire episode duration lies within `Trial` interval.

    Experimental:
    Train episode start position within interval is drawn from beta-distribution with default parameters b_alpha=1,
    b_beta=1, i.e. uniform one.

    Beta-distribution makes skewed sampling possible , e.g.
    to give recent episodes higher probability of being sampled, e.g.:  b_alpha=10, b_beta=0.8.

    It can be set to anneal to uniform one in specified number of train episodes. Annealing is done by exponentially
    decaying alpha and beta parameters to 1.

    Test episodes are always sampled uniformly.

    See description at `BTgymTrialRandomIterator()` for motivation.
    """
    trial_params = dict(
        # Trial-sampling params:
        train_range=dict(  # Trial time range in days, hours, minutes:
            days=7,
            hours=0,
        ),
        test_range=dict(  # Test time period in days, hours, minutes:
            days=7,
            hours=0,
        ),
        train_samples=0,
        test_samples=0,
        test_period=100,
        trial_start_00=True,
        expanding=False,
        b_alpha=1.0,
        b_beta=1.0,
        b_anneal_steps=-1
    )

    def __init__(self, **kwargs):
        """
        Args:
            kwargs:             BTgymDataset specific kwargs.
            train_range:        dict. containing `Trial` train interval in: `days`[, `hours`][, `minutes`];
            test_range:         dict. containing `Trial` test interval in: `days`[, `hours`][, `minutes`];
            train_samples:      number of episodes to draw from single `Trial train interval`;
            test_samples:       number of episodes to draw from `Trial test interval` every `test period`;
            test_period:        draw test episodes after every `test_period` train samples;
            expanding:          bool, if True - use expanding-type Trials, sliding otherwise; def=False;
            b_alpha:            sampling beta-distribution alpha param; def=1;
            b_beta:             sampling beta-distribution beta param; def=1;
            b_anneal_steps:     if set, anneals beta-distribution to uniform one in 'b_anneal_steps' number
                                of train samples, numbering continuously for all `Trials`; def=-1 (disabled);
            trial_start_00:     `Trial` start time will be set to that of first record of the day (usually 00:00);


        Note:
            - Total number of `Trials` (cardinality) is inferred upon args given and overall dataset size.
        """
        self.params.update(self.trial_params)
        super(BTgymSequentialTrial, self).__init__(**kwargs)

        # Timedeltas:
        self.train_range_delta = datetime.timedelta(**self.train_range)
        self.test_range_delta = datetime.timedelta(**self.test_range)

        self.train_range_row = 0
        self.test_range_row = 0
        self.train_mean_row = 0

        self.test_range_row = 0
        self.test_mean_row = 0

        self.global_step = 0
        self.total_steps = 0
        self.total_trials = 0
        self.trial_num = 0
        self.train_sample_num = 0
        self.test_sample_num = 0
        self.total_samples = 0

    @staticmethod
    def lin_decay(step, param_0, max_steps):
        """
        Linear decay from param_0 to 1 in `max_steps`.
        """
        if max_steps > 0:
            if step <= max_steps:
                return ((1 - param_0) / max_steps) * step + param_0

            else:
                return 1.0

        else:
            return param_0

    @staticmethod
    def exp_decay(step, param_0, max_steps, gamma=3.5):
        """
        For given step <= max_steps returns exp-decayed value in [param_0, 1]; returns 1 if step > max_steps;
        gamma - steepness control.
        """
        if max_steps > 0:
            if step <= max_steps:
                step = 2 - step / max_steps
                return math.exp(step ** gamma - 2 ** gamma) * (param_0 - 1) + 1

            else:
                return 1.0

        else:
            return param_0

    def sample(self, **kwargs):
        """
        Randomly samples from iterating sequence of `Trial` train/test distributions.

        Sampling loop::

            - until Trial_sequence is exhausted or .reset():
                - sample next Trial in Trial_sequence;
                    - until predefined number of episodes has been drawn:
                        - randomly draw single episode from current Trial TRAIN distribution;
                        - if reached test_period train episodes:
                            - until predefined number of episodes has been drawn:
                                - draw single episode from current Trial TEST distribution;

        Args:
            kwargs:     not used.

        Returns:
            `BTgymDataset` instance containing episode data [and metadata].
        """
        try:
            assert self.is_ready

        except AssertionError:
            return 'Data not ready. Call .reset() first.'

        episode, trial_num, type, sample_num = self._trial_sample_sequential()
        episode.metadata['type'] = type  # 0 - train, 1 - test
        episode.metadata['trial_num'] = trial_num
        episode.metadata['sample_num'] = sample_num
        self.log.debug('Seq_Data_Iterator: sample is ready with metadata: {}'.format(episode.metadata))
        return episode

    def reset(self, global_step=0, total_steps=None, skip_frame=10):
        """
        [Re]starts sampling iterator from specified position.

        Args:
            global_step:    position in [0, total_steps] interval to start sampling from.
            total_steps:    max gym environmnet steps allowed for full sweep over `Trials`.
            skip_frame:     BTGym specific, such as: `total_btgym_dataset_steps = total_steps * skip_frame`.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.read_csv()

        # Total gym-environment steps and step training starts with:
        if total_steps is not None:
            self.total_steps = total_steps
            self.global_step = global_step
            assert self.global_step < self.total_steps, 'Outer space jumps not supported. Got: global_step={} of {}.'.\
                format(self.global_step, self.total_steps)

        else:
            self.global_step = 0
            self.total_steps = -1

        # Trial train support interval in number of records:
        self.train_range_row = int( self.train_range_delta.total_seconds() / (self.timeframe * 60))

        # Trial test support interval in number of records:
        self.test_range_row = int( self.test_range_delta.total_seconds() / (self.timeframe * 60))

        # Infer cardinality of distribution over Trials:

        self.total_trials = int(
            (self.data.shape[0] - self.train_range_row) / self.test_range_row
        )

        assert self.total_trials > 0, 'Trial`s cardinality below 1. Hint: check data parameters consistency.'

        # Infer number of train samples to draw from each Trial distribution:
        if self.total_steps > 0:
            self.train_samples = int(self.total_steps / (self.total_trials * self.episode_num_records / skip_frame))

        else:
            self.log.warning('`reset_data()` got total_steps=None -> train_samples={}, iterating from 0'.
                             format(self.train_samples))

        assert self.train_samples > 0, 'Number of train samples per trial below 1. Hint: check parameters consistency.'
        assert self.test_samples >= 0, 'Size of test samples batch below 0. Hint: check parameters consistency.'

        assert self.b_alpha > 0 and self.b_beta > 0, 'Expected positive B-distribution alpha, beta; got: {}'.\
            format([self.b_alpha, self.b_beta])

        # Current trial to start with:
        self.trial_num = int(self.total_trials * self.global_step / self.total_steps)

        # Number of train samples sampled so far (fror B-distr. annealing):
        self.total_samples = self.trial_num * self.train_samples

        #print('self.train_range_delta:', self.train_range_delta.total_seconds())
        #print('self.train_range_row:', self.train_range_row)
        #print('self.test_range_delta:', self.test_range_delta)

        self.train_sample_num = 0
        self.test_sample_num = 0

        # Mean of first train-Trial:
        self.train_mean_row = int(self.train_range_row / 2) + self.test_range_row * self.trial_num
        #print('self.train_mean_row:', self.train_mean_row)

        # If trial_start_00 option set, get index of first record of that day:
        if self.trial_start_00:
            train_first_row = self.train_mean_row - int(self.train_range_row / 2) + 1
            train_first_day = self.data[train_first_row:train_first_row + 1].index[0]
            self.train_mean_row = self.data.index.get_loc(train_first_day.date(), method='nearest') + \
                                  int(self.train_range_row / 2)
            self.log.warning('Trial train start time adjusted to <00:00>')

        # Mean of first test-Trial:
        self.test_mean_row = self.train_mean_row + int((self.train_range_row + self.test_range_row) / 2) + 1
        #print('self.test_mean_row:', self.test_mean_row)

        if self.expanding:
            start_time = self.data.index[0]
            start_row = 0
            t_type='EXPANDING'

        else:
            start_time = self.data.index[self.train_mean_row - int(self.train_range_row / 2)]
            start_row = self.train_mean_row - int(self.train_range_row / 2)
            t_type = 'SLIDING'

        self.log.warning(
            (
                '\nTrial type: {}; [initial] train interval: {}; test interval: {}.' +
                '\nCardinality: {}; iterating from: {}.' +
                '\nTrain episodes per trial: {}, sampling from beta-distribution[a:{}, b:{}] on train interval.'+
                '\nSampling {} test episodes after every {} train ones.'
            ).format(
                t_type,
                self.train_range_delta,
                self.test_range_delta,
                self.total_trials,
                self.trial_num,
                self.train_samples,
                self.b_alpha,
                self.b_beta,
                self.test_samples,
                self.test_period,

            )
        )
        if self.b_anneal_steps > 0:
            self.log.warning('\nAnnealing beta-distribution to uniform one in {} train samples.'.format(self.b_anneal_steps))
        self.log.warning(
            '\nTrial #{}:\nTraining @: {} <--> {};\nTesting  @: {} <--> {}'.
            format(
                self.trial_num,
                start_time,
                self.data.index[self.train_mean_row + int(self.train_range_row / 2)],
                self.data.index[self.test_mean_row - int(self.test_range_row / 2)],
                self.data.index[self.test_mean_row + int(self.test_range_row / 2)],
            )
        )
        self.log.debug(
            'Trial #{} rows: training @: {} <--> {}; testing @: {} <--> {}'.
            format(
                self.trial_num,
                start_row,
                self.train_mean_row + int(self.train_range_row / 2),
                self.test_mean_row - int(self.test_range_row / 2),
                self.test_mean_row + int(self.test_range_row / 2),
            )
        )
        self.is_ready = True

    def _trial_sample_sequential(self):

        # Is it time to run tests?
        if self.train_sample_num != 0 and self.train_sample_num % self.test_period == 0:
            # Until not done with testing:
            if self.test_sample_num < self.test_samples:
                self.test_sample_num += 1
                self.log.debug('Test sample #{}'.format(self.test_sample_num))
                # Uniformly sample tests:
                return self._sample_interval(
                    interval=[
                        self.test_mean_row - int(self.test_range_row / 2),
                        self.test_mean_row + int(self.test_range_row / 2)
                    ],
                    b_alpha=1,
                    b_beta=1
                ), self.trial_num, True, self.test_sample_num

            else:
                self.test_sample_num = 0

        # Have we done with training on current Trial?
        if self.train_sample_num >= self.train_samples:
            self.trial_num += 1
            self.train_sample_num = 0
            self.train_mean_row += self.test_range_row
            assert self.trial_num <= self.total_trials, 'Trial`s sequence exhausted.'  # Todo: self.ready = False

            # If trial_start_00 option set, get index of first record of that day:
            if self.trial_start_00:
                train_first_row = self.train_mean_row - int(self.train_range_row / 2) + 1
                train_first_day = self.data[train_first_row:train_first_row + 1].index[0]
                self.train_mean_row = self.data.index.get_loc(train_first_day.date(), method='nearest') + \
                                      int(self.train_range_row / 2)
                self.log.debug('Trial train start time adjusted to <00:00> :{}'.format(self.train_mean_row))
            self.test_mean_row = self.train_mean_row + int((self.train_range_row + self.test_range_row) / 2) + 1

            if self.expanding:
                start_time = self.data.index[0]

            else:
                start_time = self.data.index[self.train_mean_row - int(self.train_range_row / 2)]

            self.log.warning(
                'Trial #{}:\nTraining @: {} <--> {};\nTesting  @: {} <--> {}'.
                format(
                    self.trial_num,
                    start_time,
                    self.data.index[self.train_mean_row + int(self.train_range_row / 2)],
                    self.data.index[self.test_mean_row - int(self.test_range_row / 2)],
                    self.data.index[self.test_mean_row + int(self.test_range_row / 2)],
                )
            )

        self.train_sample_num += 1
        self.total_samples += 1
        self.log.debug('Train sample #{}'.format(self.train_sample_num))

        if self.expanding:
            interval = [0, self.train_mean_row + int(self.train_range_row / 2)]

        else:
            interval = [
                self.train_mean_row - int(self.train_range_row / 2),
                self.train_mean_row + int(self.train_range_row / 2)
            ]
        return self._sample_interval(
            interval=interval,
            b_alpha=self.exp_decay(self.total_samples, self.b_alpha, self.b_anneal_steps),
            b_beta=self.exp_decay(self.total_samples, self.b_beta, self.b_anneal_steps),
        ), self.trial_num, False, self.train_sample_num


class BTgymRandomTrial(BTgymSequentialTrial):
    """
    Random Data Trials iterator.

    Note:
        While these iterators can simply be seen as sliding/random sampling time-windows, the realisation is inspired by
        `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING` paper by Duan et al.,
        https://arxiv.org/pdf/1611.02779.pdf

        Problem: Real-world BTGym POMDP violates condition of having stationary transitional distribution.

        Want: re-present BTGym on-line task as set [actually a well-ordered] of discrete-time finite-horizon discounted
        partially-observed Markov decision processes (POMDP's) to define optimization objective of learning
        RL algorithm itself, which can [hopefully] be one approach to learning in changing environment.

        Note that BTgym dataset is set of date_time ordered, mainly continuous records.
        Let each `Trial` be discrete uniform distribution among all episodes, such as for each episode:
        a) start time lies within particular `Trail support time interval` of `trial_range` time length and
        b) episode duration is less or equal to `max_episode_duration` constant;
        let `Trail mean` be particular row or date_time position within dataset timeline.

        Under original BTGym conditions, for each `Trial` there exists `single POMDP`.

        Let `Trial_sequence` be a set of `Trials` resulted by incrementing `trial_mean` parameter
        with `trial_stride` from end to end of dataset timeline.

        Such `Trial_sequence` casts a `set of POMDP's`, every element of wich can be considered well-defined in terms
        of own transition distribution; now it's possible to design optimization objective `...to maximize the expected
        total discounted reward accumulated during a single trial rather than a single episode.` [above paper, 2.2]

        This particular iterator casts unordered `set of Trials`, while `BTgymSequentialTrial()` class sweeps through
        later in time-ordered fashion.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:           BTgymDataset() specific kwargs.
            trial_range:        dict. containing `Trial support interval` (time range) as: `days`[, `hours`][, `minutes`].
            trial_stride:       dict. containing `stride interval` between `Trials` as: `days`[, `hours`][, `minutes`].
            samples_per_trial:  self-explaining; unlike sequential case, has to be set explicitly.
        """
        super(BTgymRandomTrial, self).__init__(**kwargs)
        self.trial_num = 0

    def reset(self, **kwargs):
        """
        [Re]starts sampling iterator.

        Args:
            kwargs:     not used.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            self.read_csv()

        # Infer cardinality of distribution over Trials:
        self.total_trials = int((self.data_range_delta - self.train_range_delta) / self.test_range_delta)

        assert self.total_trials > 0, 'Trial`s cardinality below 1. Hint: check data parameters consistency.'

        # Current trial to start with:
        self.trial_num = 0

        # Trial support interval in number of records:
        self.trial_range_row = int(self.data.shape[0] * (self.train_range_delta / self.data_range_delta))

        # Sequential step size:
        self.trial_stride_row = int(self.data.shape[0] * (self.test_range_delta / self.data_range_delta))

        self.sample_num = 0

        # Mean of first Trial:
        self.trial_mean_row = int(self.trial_range_row / 2) + \
                              self.trial_stride_row * int(self.total_trials * random.random())

        self.log.warning(
            '\nTrial support interval: {}; mean stride: {}\nTrials cardinality: {}\nEpisodes per trial: {}.\n'.
                format(
                self.train_range_delta,
                self.test_range_delta,
                self.total_trials,
                self.train_samples
            )
        )
        self.log.warning(
            'Trial #{} @ interval: {} <--> {}; mean row: {}'.
            format(
                self.trial_num,
                self.data.index[self.trial_mean_row - int(self.trial_range_row / 2)],
                self.data.index[self.trial_mean_row + int(self.trial_range_row / 2)],
                self.trial_mean_row
            )
        )
        self.is_ready = True

    def sample(self, **kwargs):
        """
        Randomly uniformly samples episode from `Trial` which in turn has been
        uniformly sampled from `sequence of Trials`.

        Iteratively calling this method results in::

                    - randomly draws single episode from Trial;
                - until predefined number of episodes has been drawn;
                - randomly draws Trial from Trial's distribution;
            - until bored.

        Args:
            kwargs:     not used.

        Returns:
            BTgymDataset instance containing episode data and metadata.
        """
        try:
            assert self.is_ready

        except AssertionError:
            return 'Data not ready. Call .reset() first.'

        episode = self._trial_sample_random()

        # Metadata:
        episode.metadata['type'] = False # Always `train`
        episode.metadata['trial_num'] = self.trial_num
        episode.metadata['sample_num'] = self.sample_num
        return episode

    def _trial_sample_random(self):
        if self.sample_num >= self.train_samples:
            self.trial_num += 1
            self.sample_num = 0
            self.trial_mean_row = int(self.trial_range_row / 2) +\
                                  self.trial_stride_row * int(self.total_trials * random.random())
            self.log.warning(
                'Trial #{}: from {} to {}; mean row: {}'.
                format(
                    self.trial_num,
                    self.data.index[self.trial_mean_row - int(self.trial_range_row / 2)],
                    self.data.index[self.trial_mean_row + int(self.trial_range_row / 2)],
                    self.trial_mean_row
                )
            )
        self.sample_num += 1
        self.log.debug('Trial sample #{}'.format(self.sample_num))

        return self._sample_interval(
            interval=[
                self.trial_mean_row - int(self.trial_range_row / 2),
                self.trial_mean_row + int(self.trial_range_row / 2)
            ]

        )