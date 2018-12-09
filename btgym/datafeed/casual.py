import copy
import random
import math
import datetime
from logbook import WARNING

from .base import BTgymBaseData
from .derivative import BTgymEpisode, BTgymDataTrial,  BTgymRandomDataDomain


class BTgymCasualTrial(BTgymDataTrial):
    """
    Intermediate-level data class.
    Implements conception of `Trial` object.
    Supports exact data train/test separation by means of `global_time`
    Do not use directly.
    """
    trial_params = dict(
        nested_class_ref=BTgymEpisode,
    )

    def __init__(self, name='TimeTrial', **kwargs):
        """
        Args:
            filename:           not used;
            sampling_params:    dict, sample retrieving options, see base class description for details;
            task:               int, optional;
            parsing_params:     csv parsing options, see base class description for details;
            log_level:          int, optional, logbook.level;
            _config_stack:      dict, holding configuration for nested child samples;
        """

        super(BTgymCasualTrial, self).__init__(name=name, **kwargs)
        # self.log.warning('self.frozen_time_split: {}'.format(self.frozen_time_split))

    def set_global_timestamp(self, timestamp):
        """
        Performs validity checks and sets current global_time.
        Args:
            timestamp:  POSIX timestamp

        Returns:

        """
        if self.data is not None:
            if self.frozen_split_timestamp is not None:
                self.global_timestamp = self.frozen_split_timestamp

            else:
                if self.metadata['type']:
                    if timestamp is not None:
                        assert timestamp < self.final_timestamp, \
                            'global time passed <{}> is out of upper bound <{}> for provided data.'. \
                            format(
                                datetime.datetime.fromtimestamp(timestamp),
                                datetime.datetime.fromtimestamp(self.final_timestamp)
                            )
                        if timestamp < self.start_timestamp:
                            if self.global_timestamp == 0:
                                self.global_timestamp = self.start_timestamp

                        else:
                            if timestamp > self.global_timestamp:
                                self.global_timestamp = timestamp

                    else:
                        if self.global_timestamp == 0:
                            self.global_timestamp = self.start_timestamp
                else:
                    self.global_timestamp = self.start_timestamp

    def get_global_index(self):
        """
        Returns:
            data row corresponded to current global_time
        """
        if self.is_ready:
            return self.data.index.get_loc(
                datetime.datetime.fromtimestamp(self.global_timestamp),
                method='backfill'
            )

        else:
            return 0

    def get_intervals(self):
        """
        Estimates exact sampling intervals such as test episode starts as close to current global time point as
        data consistency allows but no earlier;

        Returns:
            dict of train and test sampling intervals for current global_time point
        """
        if self.is_ready:
            if self.metadata['type']:
                # Intervals for target trial:
                current_index = self.get_global_index()

                self.log.debug(
                    'current_index: {}, total_num_records: {}, sample_num_records: {}'.format(
                        current_index,
                        self.total_num_records,
                        self.sample_num_records
                    )
                )
                assert 0 <= current_index <= self.total_num_records - self.sample_num_records,\
                    'global_time: {} outside data interval: {} - {}, considering sample duration: {}'.format(
                        self.data.index[current_index],
                        self.data.index[0],
                        self.data.index[-1],
                        self.max_sample_len_delta
                    )
                train_interval = [0, current_index]
                test_interval = [current_index + 1, self.total_num_records - 1]

            else:
                # Intervals for source trial:
                train_interval = [0, self.train_num_records - 1]
                test_interval = [self.train_num_records, self.total_num_records - 1]  # TODO: ?!

            self.log.debug(
                'train_interval: {}, datetimes: {} - {}'.
                format(
                    train_interval,
                    self.data.index[train_interval[0]],
                    self.data.index[train_interval[-1]],
                )
            )
            self.log.debug(
                'test_interval: {}, datetimes: {} - {}'.
                format(
                    test_interval,
                    self.data.index[test_interval[0]],
                    self.data.index[test_interval[-1]],
                )
            )
        else:
            train_interval = None
            test_interval = None

        return train_interval, test_interval

    def sample(
            self,
            get_new=True,
            sample_type=0,
            timestamp=None,
            align_left=True,
            b_alpha=1.0,
            b_beta=1.0,
            **kwargs
    ):
        """
        Samples continuous subset of data.

        Args:
            get_new (bool):                 sample new (True) or reuse (False) last made sample;
            sample_type (int or bool):      0 (train) or 1 (test) - get sample from train or test data subsets
                                            respectively.
            timestamp:                      POSIX timestamp.
            align_left:                     bool, if True: set test interval as close to current timepoint as possible.
            b_alpha (float):                beta-distribution sampling alpha > 0, valid for train episodes.
            b_beta (float):                 beta-distribution sampling beta > 0, valid for train episodes.
        """
        try:
            assert self.is_ready

        except AssertionError:
            self.log.exception(
                'Sampling attempt: data not ready. Hint: forgot to call data.reset()?'
            )
            raise AssertionError

        try:
            assert sample_type in [0, 1]

        except AssertionError:
            self.log.exception(
                'Sampling attempt: expected sample type be in {}, got: {}'.\
                format([0, 1], sample_type)
            )
            raise AssertionError

        # Set actual time:
        if timestamp is not None:
            self.set_global_timestamp(timestamp)

        if 'interval' not in kwargs.keys():
            train_interval, test_interval = self.get_intervals()
        else:
            train_interval = test_interval = kwargs.pop('interval')

        if self.sample_instance is None or get_new:
            if sample_type == 0:
                # Get beta_distributed sample in train interval:
                self.sample_instance = self._sample_interval(
                    train_interval,
                    b_alpha=b_alpha,
                    b_beta=b_beta,
                    name='train_' + self.sample_name,
                    **kwargs
                )

            else:
                # If parent is target - get left-aligned (i.e. as close as possible to current global_time)
                # sample in test interval; else (parenet is source) - uniformly sample from test interval:
                if self.metadata['parent_sample_type']:
                    align = align_left

                else:
                    align = False

                self.sample_instance = self._sample_aligned_interval(
                    test_interval,
                    align_left=align,
                    b_alpha=1,
                    b_beta=1,
                    name='test_' + self.sample_name,
                    **kwargs
                )
            self.sample_instance.metadata['type'] = sample_type
            self.sample_instance.metadata['sample_num'] = self.sample_num
            self.sample_instance.metadata['parent_sample_num'] = copy.deepcopy(self.metadata['sample_num'])
            self.sample_instance.metadata['parent_sample_type'] = copy.deepcopy(self.metadata['type'])
            self.sample_num += 1

        else:
            # Do nothing:
            self.log.debug('Reusing sample, id: {}'.format(self.sample_instance.filename))

        return self.sample_instance


class BTgymCasualDataDomain(BTgymRandomDataDomain):
    """
    Imitates online data stream by implementing conception of sliding `current time point`
    and enabling sampling control according to it.

    Objective is to enable proper train/evaluation/test data split and prevent data leakage by
     allowing training on known, past data only and testing on unknown, future data, providing realistic training cycle.

    Source trials set is defined as all trials starting somewhere in past and ending no later than current time point,
    and target trials set as set of trials such as: trial test period starts somewhere in the past and ends at
    current time point and trial test period starts from now on for all time points in available dataset range.

    Sampling control is defined by:
    - `current time point` is set arbitrary and is stateful in sense it can be only increased (no backward time);
    - source trials can be sampled from past (known) data multiply times;
    - target trial can only be sampled once according to current time point or later (unknown data);
    - as any sampled target trial is being evaluated by outer algorithm, current time should be incremented either by
      providing 'timestamp' arg. to sample() method or calling set_global_timestamp() method,
      to match last evaluated record (marking all evaluated data as already known
      and making it available for training);
    """
    trial_class_ref = BTgymCasualTrial
    episode_class_ref = BTgymEpisode

    def __init__(
            self,
            filename,
            trial_params,
            episode_params,
            frozen_time_split=None,
            name='TimeDataDomain',
            data_names=('default_asset',),
            **kwargs):
        """
        Args:
            filename:           Str or list of str, file_names containing CSV historic data;
            parsing_params:     csv parsing options, see base class description for details;
            trial_params:       dict, describes trial parameters, should contain keys:
                                {sample_duration, time_gap, start_00, start_weekdays, test_period, expanding};
            episode_params:     dict, describes episode parameters, should contain keys:
                                {sample_duration, time_gap, start_00, start_weekdays};
            name:               str, optional
            task:               int, optional
            log_level:          int, logbook.level

        """
        self.train_range_row = 0
        self.test_range_row = 0

        self.test_range_row = 0
        self.test_mean_row = 0

        self.global_step = 0
        self.total_samples = -1
        self.sample_num = -1
        self.sample_stride = -1

        # if frozen_time_split is not None:
        #     self.frozen_time_split = datetime.datetime(**frozen_time_split)
        #
        # else:
        #     self.frozen_time_split = None
        #
        # self.frozen_split_timestamp = None

        kwargs.update({'target_period': episode_params['sample_duration']})

        trial_params['start_00'] = False
        trial_params['frozen_time_split'] = frozen_time_split

        super(BTgymCasualDataDomain, self).__init__(
            filename=filename,
            trial_params=trial_params,
            episode_params=episode_params,
            use_target_backshift=False,
            name=name,
            data_names=data_names,
            frozen_time_split=frozen_time_split,
            **kwargs
        )

        # self.log.warning('2: self.frozen_time_split: {}'.format(self.frozen_time_split))

        self.log.debug('trial_class_ref: {}'.format(self.trial_class_ref))
        self.log.debug('episode_class_ref: {}'.format(self.episode_class_ref))

        self.log.debug('sampling_params: {}'.format(self.sampling_params))
        self.log.debug('nested_params: {}'.format(self.nested_params))

    def set_global_timestamp(self, timestamp):
        """
        Performs validity checks and sets current global_time.
        Args:
            timestamp:  POSIX timestamp

        Returns:

        """
        if self.data is not None:
            if self.frozen_split_timestamp is not None:
                self.global_timestamp = self.frozen_split_timestamp

            else:
                if timestamp is not None:
                    assert timestamp < self.final_timestamp, \
                        'global time passed <{}> is out of upper bound <{}> for provided data.'. \
                        format(
                            datetime.datetime.fromtimestamp(timestamp),
                            datetime.datetime.fromtimestamp(self.final_timestamp)
                        )
                    if timestamp < self.start_timestamp:
                        if self.global_timestamp == 0:
                            self.global_timestamp = self.start_timestamp

                    else:
                        if timestamp > self.global_timestamp:
                            self.global_timestamp = timestamp

                else:
                    if self.global_timestamp == 0:
                        self.global_timestamp = self.start_timestamp

    def get_global_index(self):
        """
        Returns:
            data row corresponded to current global_time
        """
        if self.is_ready:
            return self.data.index.get_loc(
                datetime.datetime.fromtimestamp(self.global_timestamp),
                method='backfill'
            )

        else:
            return 0

    def get_intervals(self):
        """
        Estimates exact sampling intervals such as train period of target trial overlaps by known up to date data

        Returns:
            dict of train and test sampling intervals for current global_time point
        """
        if self.is_ready:
            current_index = self.get_global_index()
            assert current_index >= self.train_num_records
            assert current_index + self.test_num_records <= self.total_num_records, 'End of data!'

            self.log.debug(
                'current_index: {}, total_num_records: {}, sample_num_records: {}'.format(
                    current_index,
                    self.total_num_records,
                    self.sample_num_records
                )
            )

            if self.expanding:
                train_interval = [0, current_index]

            else:
                train_interval = [current_index - self.sample_num_records, current_index]

            test_interval = [current_index - self.train_num_records, current_index + self.test_num_records]

            self.log.debug(
                'train_interval: {}, datetimes: {} - {}'.
                format(
                    train_interval,
                    self.data.index[train_interval[0]],
                    self.data.index[train_interval[-1]],
                )
            )
            self.log.debug(
                'test_interval: {}, datetimes: {} - {}'.
                    format(
                        test_interval,
                        self.data.index[test_interval[0]],
                        self.data.index[test_interval[-1]],
                )
            )
        else:
            train_interval = None
            test_interval = None

        return train_interval, test_interval

    def _reset(self, data_filename=None, timestamp=None, **kwargs):

        self.read_csv(data_filename)

        # Maximum data time gap allowed within sample as pydatetimedelta obj:
        self.max_time_gap = datetime.timedelta(**self.time_gap)

        # Max. gap number of records:
        self.max_gap_num_records = int(self.max_time_gap.total_seconds() / (60 * self.timeframe))

        # ... maximum episode time duration:
        self.max_sample_len_delta = datetime.timedelta(**self.sample_duration)

        # Maximum possible number of data records (rows) within episode:
        self.sample_num_records = int(self.max_sample_len_delta.total_seconds() / (60 * self.timeframe))

        self.log.debug('sample_num_records: {}'.format(self.sample_num_records))
        self.log.debug('sliding_test_period: {}'.format(self.test_period))

        # Train/test timedeltas:
        self.test_range_delta = datetime.timedelta(**self.test_period)
        self.train_range_delta = datetime.timedelta(**self.sample_duration) - datetime.timedelta(**self.test_period)

        self.test_num_records = round(self.test_range_delta.total_seconds() / (60 * self.timeframe))
        self.train_num_records = self.sample_num_records - self.test_num_records

        self.log.debug('test_num_records: {}'.format(self.test_num_records))
        self.log.debug('train_num_records: {}'.format(self.train_num_records))

        self.start_timestamp = self.data.index[self.sample_num_records].timestamp()
        self.final_timestamp = self.data.index[-self.test_num_records].timestamp()

        if self.frozen_time_split is not None:
            frozen_index = self.data.index.get_loc(self.frozen_time_split, method='ffill')
            self.frozen_split_timestamp = self.data.index[frozen_index].timestamp()
            self.set_global_timestamp(self.frozen_split_timestamp)

        else:
            self.frozen_split_timestamp = None
            self.set_global_timestamp(timestamp)
        current_index = self.get_global_index()

        try:
            assert self.train_num_records >= self.test_num_records

        except AssertionError:
            self.log.exception(
                'Train subset should contain at least one episode, got: train_set size: {} rows, episode_size: {} rows'.
                    format(self.train_num_records, self.test_num_records)
            )
            raise AssertionError

        self.sample_num = 0

        self.is_ready = True

    def sample(self, get_new=True, sample_type=0, timestamp=None, b_alpha=1.0, b_beta=1.0, **kwargs):
        """
        Samples from sequence of `Trials`.

        Args:
            get_new (bool):                 sample new (True) or reuse (False) last made sample; n/a for target trials
            sample_type (int or bool):      0 (train) or 1 (test) - get sample from source or target data subsets
                                            respectively;
            timestamp:                      POSIX timestamp indicating current global time of training loop
            b_alpha (float):                beta-distribution sampling alpha > 0, valid for train episodes.
            b_beta (float):                 beta-distribution sampling beta > 0, valid for train episodes.


        Returns:
            Trial as `BTgymBaseDataTrial` instance;
            None, if trial's sequence is exhausted (global time is up).
        """
        self.set_global_timestamp(timestamp)

        if 'interval' not in kwargs.keys():
            train_interval, test_interval = self.get_intervals()
        else:
            train_interval = test_interval = kwargs.pop('interval')

        if get_new or self.sample_instance is None:
            if sample_type:
                self.sample_instance = self._sample_interval(
                    interval=test_interval,
                    b_alpha=b_alpha,
                    b_beta=b_beta,
                    name='target_trial_',
                    **kwargs
                )
                if self.sample_instance is None:
                    # Exhausted:
                    return False

            else:
                self.sample_instance = self._sample_interval(
                    interval=train_interval,
                    b_alpha=b_alpha,
                    b_beta=b_beta,
                    name='source_trial_',
                    **kwargs
                )
                if self.sample_instance is None:
                    # Exhausted:
                    return False

            self.log.debug(
                'sampled new trial <{}> with metadata: {}'.
                format(self.sample_instance.filename, self.sample_instance.metadata)
            )

        else:
            self.log.debug(
                'reused trial <{}> with metadata: {}'.
                    format(self.sample_instance.filename, self.sample_instance.metadata)
            )
        self.sample_instance.metadata['type'] = sample_type
        self.sample_instance.metadata['sample_num'] = self.sample_num
        self.sample_instance.metadata['parent_sample_num'] = copy.deepcopy(self.metadata['sample_num'])
        self.sample_instance.metadata['parent_sample_type'] = copy.deepcopy(self.metadata['type'])

        return self.sample_instance
