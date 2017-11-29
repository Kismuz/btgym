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
            **kwargs:
                filename=None,  # Str or list of str, should be given either here  or when calling read_csv()

                Default parameters for source-specific CSV datafeed class,
                correctly parses 1 minute Forex generic ASCII
                data files from www.HistData.com:

                CSV to Pandas params:

                    sep=';',
                    header=0,
                    index_col=0,
                    parse_dates=True,
                    names=['open', 'high', 'low', 'close', 'volume'],

                Pandas to BT.feeds params:

                    timeframe=1,  # 1 minute.
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=-1,
                    openinterest=-1,

                Random-sampling params:

                    start_weekdays=[0, 1, 2, 3, ],  # Only weekdays from the list will be used for episode start.
                    start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).
                    episode_len_days=1,  # Maximum episode time duration in days, hours, minutes:
                    episode_len_hours=23,
                    episode_len_minutes=55,
                    time_gap_days=0,  # Maximum data time gap allowed within sample in days, hours. Thereby,
                    time_gap_hours=5,  # if set to be < 1 day,
                        samples containing weekends and holidays gaps will be rejected.
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
        episode.metadata['type'] = 'random_sample'
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
            while not episode_first_day.weekday() in self.start_weekdays:
                self.log.debug('Not a good day to start, resampling...')
                first_row = int((self.data.shape[0] - self.episode_num_records - 1) * random.random())
                episode_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug('Episode start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))
                attempts +=1

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

    def _sample_position(self, position, tolerance=10):
        """
        Samples continuous subset of data, starting from given 'position' with allowed `tolerance`.

        Args:
            position:   position of record to start subset from.
            tolerance:  actual start position is uniformly sampled from [position - tolerance, position + tolerance]

        Returns:
             - BTgymDataset instance such as: number of records ~ max_episode_len,
                where record number tolerance is inferred from `time_gap` param;
                first data row = `position` +- tolerance
             - `False` if it is not possible to sample instance with set args.
        """
        try:
            assert not self.data.empty

        except (AssertionError, AttributeError) as e:
            raise  AssertionError('BTgymDataset instance holds no data. Hint: forgot to call .read_csv()?')

        sample_num_records = self.episode_num_records

        try:
            assert tolerance / 2 < position\
                   < int((self.data.shape[0] - sample_num_records - int(tolerance / 2) - 1))

        except AssertionError:
            self.log.warning('Cannot sample with size {}, starting from position {}+-{} from dataset of {} records'.
                             format(sample_num_records, position, tolerance, self.data.shape[0]))
            return False

        self.log.debug('Maximum sample time duration set to: {}.'.format(self.max_episode_len))
        self.log.debug('Respective number of steps: {}.'.format(sample_num_records))
        self.log.debug('Maximum allowed data time gap set to: {}.\n'.format(self.max_time_gap))

        # Sanity check param:
        max_attempts = 100
        attempts = 0

        # # Keep sampling random enter points until all conditions are met:
        while attempts <= max_attempts:
            first_row = position - tolerance + round(2 * tolerance * random.random())
            episode_first_day = self.data[first_row:first_row + 1].index[0]
            self.log.debug('Sample start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))

            # Keep sampling until good day:
            while not episode_first_day.weekday() in self.start_weekdays:
                self.log.debug('Not a good day to start, resampling...')
                first_row = position - tolerance + round(2 * tolerance * random.random())
                episode_first_day = self.data[first_row:first_row + 1].index[0]
                self.log.debug('Sample start: {}, weekday: {}.'.format(episode_first_day, episode_first_day.weekday()))
                attempts += 1

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
                episode.filename = '_btgym_position_sample_' + str(adj_timedate)
                self.log.info('Sample id: <{}>.'.format(episode.filename))
                episode.data = episode_sample
                episode.metadata['type'] = 'position_sample'
                episode.metadata['first_row'] = first_row
                return episode

            else:
                self.log.debug('Duration too big, resampling...\n')
                attempts += 1

        # Got here -> sanity check failed:
        msg = ('Quitting after {} sampling attempts.' +
               'Hint: check sampling params / dataset consistency.').format(attempts)
        self.log.warning(msg)
        return False


class BTgymSequentialTrial(BTgymDataset):
    """
    Sequential Data Trials iterator.

    See `Notes` at `BTgymTrialRandomIterator()` for description and motivation.

    For this class, `Trials` are sampled in ordered `sliding timewindow` rather than random fashion.
    """
    trial_params =dict(
        # Trial-sampling params:
        trial_range=dict(  # Trial time range in days, hours, minutes:
            days=7,
            hours=0,
        ),
        trial_stride=dict(
            days=7,
            hours=0,
        ),
        samples_per_trial=0
    )

    def __init__(self, **kwargs):
        """
        Args:
            kwargs:         BTgymDataset specific kwargs.
            trial_range:    dict. containing `Trial` support interval (time range) in: `days`[, `hours`][, `minutes`].
            trial_stride:   dict. containing stride interval between `Trials` in: `days`[, `hours`][, `minutes`].

        Note:
            - Total number of `Trials` is inferred upon trial_params given and overall dataset size.

            - Total number of episodes drawn from each `Trial` is defined when own .reset() method is called.
        """
        self.params.update(self.trial_params)
        super(BTgymSequentialTrial, self).__init__(**kwargs)

        # Timedeltas:
        self.trial_range_delta = datetime.timedelta(**self.trial_range)
        self.trial_stride_delta = datetime.timedelta(**self.trial_stride)

        self.trial_range_row = 0
        self.trial_stride_row = 0
        self.trial_mean_row = 0
        self.global_step = 0
        self.total_steps = 0
        self.total_trials = 0
        self.trial_num = 0
        self.sample_num = 0

    def sample(self, **kwargs):
        """
        Randomly uniformly samples from iterating sequence of `Trial` distributions.

        Iteratively calling this method results in::

                    - randomly draws single episode from first [or specified by `reset()`] Trial;
                - until predefined number of episodes has been drawn;
                - advances to the next Trial in Trial_sequence;
            - until Trial_sequence is exhausted or .reset();

        Args:
            kwargs:     not used.

        Returns:
            BTgymDataset instance containing episode data and metadata.
        """
        try:
            assert self.is_ready

        except AssertionError:
            return 'Data not ready. Call .reset() first.'

        episode = self._trial_sample_sequential()
        episode.metadata['type'] = 'sequential_trial_sample'
        episode.metadata['trial_num'] = self.trial_num
        episode.metadata['sample_num'] = self.sample_num
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

        # Infer cardinality of distribution over Trials:
        self.total_trials = int((self.data_range_delta - self.trial_range_delta) / self.trial_stride_delta)

        assert self.total_trials > 0, 'Trial`s cardinality below 1. Hint: check data parameters consistency.'

        # Number of samples to draw from each Trial distribution:
        if self.total_steps > 0:
            self.samples_per_trial = int(self.total_steps / (self.total_trials * self.episode_num_records / skip_frame))

        else:
            self.log.warning('`reset_data()` got total_steps=None -> samples_per_trial={}, iterating from 0'.
                             format(self.samples_per_trial))

        assert self.samples_per_trial > 0, 'Number of samples per trial below 1. Hint: check parameters consistency.'

        # Current trial to start with:
        self.trial_num = int(self.total_trials * self.global_step / self.total_steps)

        # Trial support interval in number of records:
        self.trial_range_row = int(self.data.shape[0] * (self.trial_range_delta / self.data_range_delta))

        # Sequential step size:
        self.trial_stride_row = int(self.data.shape[0] * (self.trial_stride_delta / self.data_range_delta))

        self.sample_num = 0

        # Mean of first Trial:
        self.trial_mean_row = int(self.trial_range_row / 2) + self.trial_stride_row * self.trial_num

        self.log.warning(
            '\nTrial interval: {}; stride: {}.\nTrials cardinality: {}; iterating from: {}.\nEpisodes per trial: {}.\n'.
            format(
                self.trial_range_delta,
                self.trial_stride_delta,
                self.total_trials,
                self.trial_num,
                self.samples_per_trial
            )
        )
        self.log.warning(
            'Trial #{} @ interval: {} <--> {}'.
            format(
                self.trial_num,
                self.data.index[self.trial_mean_row - int(self.trial_range_row / 2)],
                self.data.index[self.trial_mean_row + int(self.trial_range_row / 2)]
            )
        )
        self.is_ready = True

    def _trial_sample_sequential(self):
        if self.sample_num >= self.samples_per_trial:
            self.trial_num += 1
            self.sample_num = 0
            self.trial_mean_row += self.trial_stride_row

            assert self.trial_num <= self.total_trials, 'Trial`s sequence exhausted.'
            # Todo: self.ready = False

            self.log.warning(
                'Trial #{}: from {} to {}'.
                format(
                    self.trial_num,
                    self.data.index[self.trial_mean_row - int(self.trial_range_row / 2)],
                    self.data.index[self.trial_mean_row + int(self.trial_range_row / 2)]
                )
            )
        self.sample_num += 1
        self.log.debug('Trial sample #{}'.format(self.sample_num))

        return self._sample_position(position=self.trial_mean_row, tolerance=int(self.trial_range_row / 2))


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
        self.total_trials = int((self.data_range_delta - self.trial_range_delta) / self.trial_stride_delta)

        assert self.total_trials > 0, 'Trial`s cardinality below 1. Hint: check data parameters consistency.'

        # Current trial to start with:
        self.trial_num = 0

        # Trial support interval in number of records:
        self.trial_range_row = int(self.data.shape[0] * (self.trial_range_delta / self.data_range_delta))

        # Sequential step size:
        self.trial_stride_row = int(self.data.shape[0] * (self.trial_stride_delta / self.data_range_delta))

        self.sample_num = 0

        # Mean of first Trial:
        self.trial_mean_row = int(self.trial_range_row / 2) + \
                              self.trial_stride_row * int(self.total_trials * random.random())

        self.log.warning(
            '\nTrial support interval: {}; mean stride: {}\nTrials cardinality: {}\nEpisodes per trial: {}.\n'.
                format(
                self.trial_range_delta,
                self.trial_stride_delta,
                self.total_trials,
                self.samples_per_trial
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
        episode.metadata['type'] = 'random_trial_sample'
        episode.metadata['trial_num'] = self.trial_num
        episode.metadata['sample_num'] = self.sample_num
        return episode

    def _trial_sample_random(self):
        if self.sample_num >= self.samples_per_trial:
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

        return self._sample_position(position=self.trial_mean_row, tolerance=int(self.trial_range_row / 2))