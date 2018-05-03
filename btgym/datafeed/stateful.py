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

import random
import math
import datetime

from .derivative import BTgymRandomDataDomain


class BTgymSequentialDataDomain(BTgymRandomDataDomain):
    """
    Top-level sequential data iterator.
    Implements conception of sliding [or expanding] train/test time-window.
    Due to sequential nature doesnt support firm source/target domain separation.

    Note:

        Single Trial is defined by support interval (overall duration)  and test interval::

            [trial_start_time=train_start_time <-> train_end_time=test_start_time <-> test_end_time=trial_end_time],

        Sliding time-window data iterating:

        If training is started from the beginningg of the dataset, `train_start_time` is set to that of first record,
        for example, for the start of the year::

            Trial duration: 10 days; test interval: 2 days, iterating from 0-th, than:

            train interval: 8 days, 0:00:00; test interval: 2 days, 0:00:00.

        Then first trial intervals will be (note that omitted data periods like holidays will not be counted,
        so overall trial duration is dilated to get proper number of records)::

            Trial #0 @: 2016-01-03 17:01:00 <--> 2016-01-17 17:05:00,
            and last two days will be reserved for test data

        Since `reset_data()` method call, every next call to `BTgymSequentialDataDomain.sample()` method will return
        Trial object, such as::

            train_start_time_next_trial = `test_end_time_previous_trial + 1_time_unit

        i.e. next trial will be shifted by the duration of test period.

        Repeats until entire dataset is exhausted.

        Note that while train periods are overlapping, test periods form a partition.

        Here, next trial will be shifted by two days::

            Trial #1 @: 2016-01-05 00:00:00 <--> 2016-01-19 00:10:00


        Expanding time-window data iterating:

        Differs from above in a way that trial interval start position is fixed at the earliest time of dataset. Thus,
        trial support interval is expanding to the right and every subsequent trial is `longer` than previous one
        by amount of test interval.

        Episodes sampling:

        Episodes sampling is performed in such a way that entire episode duration lies within `Trial` interval.

    """

    @staticmethod
    def _lin_decay(step, param_0, max_steps):
        # TODO: move to utils
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
    def _exp_decay(step, param_0, max_steps, gamma=3.5):
        # TODO: not used here anymore, move to utils
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

    def __init__(self, name='SeqDataDomain', **kwargs):
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

        Note:
            - Total number of `Trials` (cardinality) is inferred upon args given and overall dataset size.
        """
        self.train_range_row = 0
        self.test_range_row = 0

        self.test_range_row = 0
        self.test_mean_row = 0

        self.global_step = 0
        self.total_samples = -1
        self.sample_num = -1
        self.sample_stride = -1

        super(BTgymSequentialDataDomain, self).__init__(name=name, **kwargs)

    def sample(self, **kwargs):
        """
        Iteratively samples from sequence of `Trials`.

        Sampling loop::

            - until Trial_sequence is exhausted or .reset():
                - sample next Trial in Trial_sequence;

        Args:
            kwargs:             not used.

        Returns:
            Trial as `BTgymBaseDataTrial` instance;
            None, if trial's sequence is exhausted.
        """
        self.sample_instance = self._sample_sequential()
        if self.sample_instance is None:
            # Exhausted:
            return False

        else:
            self.sample_instance.metadata['type'] = 0  # 0 - always train
            self.sample_instance.metadata['sample_num'] = self.sample_num
            self.log.debug(
                'got new trial <{}> with metadata: {}'.
                format(self.sample_instance.filename, self.sample_instance.metadata)
            )
            return self.sample_instance

    def _get_interval(self, sample_num):
        """
        Defines exact interval and corresponding datetime stamps for Trial
        Args:
            sample_num: Trial position in iteration sequence

        Returns:
            two lists: [first_row, last_row], [start_time, end_time]
        """
        # First current trial interval:
        first_row = sample_num * self.sample_stride

        if self.start_00:
            first_day = self.data[first_row:first_row + 1].index[0]
            first_row = self.data.index.get_loc(first_day.date(), method='nearest')
            self.log.debug('Trial train start time adjusted to <00:00>')

        last_row = first_row + self.sample_num_records

        if self.expanding:
            start_time = self.data.index[0]
            first_row = 0

        else:
            start_time = self.data.index[first_row]

        end_time = self.data.index[last_row]
        return [first_row, last_row], [start_time, end_time]

    def reset(self, global_step=0, total_steps=None, skip_frame=10, data_filename=None, **kwargs):
        """
        [Re]starts sampling iterator from specified position.

        Args:
            data_filename:  Str or list of str, file_names containing CSV historic data;
            global_step:    position in [0, total_steps] interval to start sampling from;
            total_steps:    max gym environmnet steps allowed for full sweep over `Trials`;
            skip_frame:     BTGym specific, such as: `total_btgym_dataset_steps = total_steps * skip_frame`;
        """
        self._reset(data_filename=data_filename, **kwargs)

        # Total gym-environment steps and step training starts with:
        if total_steps is not None:
            self.total_steps = total_steps
            self.global_step = global_step
            try:
                assert self.global_step < self.total_steps

            except AssertionError:
                self.log.exception(
                    'Outer space jumps not supported. Got: global_step={} of {}.'.
                    format(self.global_step, self.total_steps)
                )
                raise AssertionError

        else:
            self.global_step = 0
            self.total_steps = -1

        # Infer Trial test support interval in number of records:
        self.trial_test_range_delta = datetime.timedelta(**self.nested_params['sampling_params']['test_period'])
        self.trial_test_range_row = int(self.trial_test_range_delta.total_seconds() / (self.timeframe * 60))

        #print('self.trial_test_range_delta.total_seconds(): ', self.trial_test_range_delta.total_seconds())
        #print('self.trial_test_range_row: ', self.trial_test_range_row)

        # Infer Trial train support interval in number of records:
        self.trial_train_range_delta = self.max_sample_len_delta - self.trial_test_range_delta

        try:
            assert self.trial_train_range_delta.total_seconds() > 0

        except AssertionError:
            self.log.exception(
                'Trial train period should not be negative, got: {}'.format(self.trial_train_range_delta)
            )
            raise AssertionError

        self.trial_train_range_row = int(self.trial_train_range_delta.total_seconds() / (self.timeframe * 60))

        # Infer cardinality of Trials:
        self.total_samples = int(
            (self.data.shape[0] - self.trial_train_range_row) / self.trial_test_range_row
        )

        # Set domain sample stride as duration of Trial test period:
        self.sample_stride = self.trial_test_range_row

        try:
            assert self.total_samples > 0

        except AssertionError:
            self.log.exception(
                'Trial`s cardinality below 1. Hint: check data parameters consistency.'
            )
            raise AssertionError

        # Current trial to start with:
        self.sample_num = int(self.total_samples * self.global_step / self.total_steps)

        if self.expanding:
            t_type = 'EXPANDING'

        else:
            t_type = 'SLIDING'

        self.log.notice(
            (
                '\nTrial type: {}; [initial] train interval: {}; test interval: {}.' +
                '\nCardinality: {}; iterating from: {}.'
            ).format(
                t_type,
                self.trial_train_range_delta,
                self.trial_test_range_delta,
                self.total_samples,
                self.sample_num,
            )
        )

        self.is_ready = True

    def _sample_sequential(self):
        """
        Iteratively samples Trials.
        Returns:
                (Trial instance, trial number)
                (None, trials_cardinality),  it trials sequence exhausted
        """

        if self.sample_num > self.total_samples:
            self.is_ready = False
            self.log.warning('Sampling sequence exhausted at {}-th Trial'.format(self.sample_num))
            return None

        else:
            # Get Trial:
            interval, time = self._get_interval(self.sample_num)

            self.log.notice(
                'Trial #{} @: {} <--> {};'.format(self.sample_num, time[0], time[-1])
            )
            self.log.debug(
                'Trial #{} rows: {} <--> {}'.
                    format(
                    self.sample_num,
                    interval[0],
                    interval[-1]
                )
            )
            trial = self._sample_interval(interval, name='sequential_trial_')
            self.sample_num += 1
            return trial


