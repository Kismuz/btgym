###############################################################################
#
# Copyright (C) 2017-2018 Andrew Muzikin
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

from logbook import WARNING
from .base import BTgymBaseData
import datetime


class BTgymEpisode(BTgymBaseData):
    """
    Low-level data class.
    Implements `Episode` object containing single episode data sequence.
    Doesnt allows further sampling and data loading.
    Supposed to be converted to bt.datafeed object via .to_btfeed() method.
    Do not use directly.
    """
    def __init__(
            self,
            dataframe,
            parsing_params=None,
            sampling_params=None,
            name=None,
            data_names=('default_asset',),
            task=0,
            log_level=WARNING,
            _config_stack=None,
    ):

        super(BTgymEpisode, self).__init__(
            dataframe=dataframe,
            parsing_params=parsing_params,
            sampling_params=None,
            name='episode',
            task=task,
            data_names=data_names,
            log_level=log_level,
            _config_stack=_config_stack
        )

    def reset(self, **kwargs):
        raise RuntimeError('Episode object doesnt support .reset() method.')

    def sample(self, **kwargs):
        raise RuntimeError('Episode object doesnt support .sample() method.')


class BTgymDataTrial(BTgymBaseData):
    """
    Intermediate-level data class.
    Implements conception of `Trial` object.
    Supports data train/test separation.
    Do not use directly.
    """
    trial_params = dict(
        nested_class_ref=BTgymEpisode,
    )

    def __init__(
            self,
            dataframe,
            parsing_params=None,
            sampling_params=None,
            name=None,
            data_names=('default_asset',),
            frozen_time_split=None,
            task=0,
            log_level=WARNING,
            _config_stack=None,


    ):
        """
        Args:
            filename:           not used;
            sampling_params:    dict, sample retrieving options, see base class description for details;
            task:               int, optional;
            parsing_params:     csv parsing options, see base class description for details;
            log_level:          int, optional, logbook.level;
            _config_stack:      dict, holding configuration for nested child samples;
        """

        super(BTgymDataTrial, self).__init__(
            dataframe=dataframe,
            parsing_params=parsing_params,
            sampling_params=sampling_params,
            name='Trial',
            data_names=data_names,
            frozen_time_split=frozen_time_split,
            task=task,
            log_level=log_level,
            _config_stack=_config_stack
        )


class BTgymRandomDataDomain(BTgymBaseData):
    """
    Top-level data class. Implements one way data domains can be defined,
    namely when source domain precedes and target one. Implements pipe::

        Domain.sample() --> Trial.sample() --> Episode.to_btfeed() --> bt.Startegy

    This particular class randomly samples Trials from provided dataset.

    """
    # Classes to use for sample objects:
    trial_class_ref = BTgymDataTrial
    episode_class_ref = BTgymEpisode

    def __init__(
            self,
            trial_params,
            episode_params,
            dataframe,
            parsing_params=None,
            target_period=None,
            use_target_backshift=False,
            frozen_time_split=None,
            name='RndDataDomain',
            task=0,
            data_names=('default_asset',),
            log_level=WARNING,
    ):
        """
        Args:
            dataframe:          pd.dataframe or iterable of pd.dataframes containing historic data;
            parsing_params:         csv parsing options, see base class description for details;
            trial_params:           dict, describes trial parameters, should contain keys:
                                    {sample_duration, time_gap, start_00, start_weekdays, test_period, expanding};
            episode_params:         dict, describes episode parameters, should contain keys:
                                    {sample_duration, time_gap, start_00, start_weekdays};

            target_period:          dict, None or Int, domain target period, def={'days': 0, 'hours': 0, 'minutes': 0};
                                    setting this param to non-zero duration forces separation to source/target
                                    domains (which can be thought of as creating  top-level train/test subsets) with
                                    target data duration equal to `target_period`;
                                    if set to None - no target period assumed;
                                    if set to -1 - no source period assumed;
                                    Source data always precedes target one.
            use_target_backshift:   bool, if true - target domain is shifted back by the duration of trial train period,
                                    thus allowing training on part of target domain data,
                                    namely train part of the trial closest to source/target break point.
            name:                   str, optional
            task:                   int, optional
            log_level:              int, logbook.level
        """
        sample_params_keys = {'sample_duration', 'time_gap'}

        assert isinstance(trial_params, dict) and sample_params_keys <= set(trial_params.keys()),\
            'Expected dict. <trial_params> contain keys: {}, got: {}'.format(sample_params_keys, trial_params)

        assert isinstance(episode_params, dict) and sample_params_keys <= set(episode_params.keys()), \
            'Expected dict. <episode_params> contain keys: {}, got: {}'.format(sample_params_keys, episode_params)

        if parsing_params is None:
            parsing_params = dict(
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

        # Hacky cause we want trial test period to be attr of Trial instance
        # and top-level test (target) period to be attribute of Domain instance:
        try:
            trial_test_period = trial_params.pop('test_period')

        except(AttributeError, KeyError):
            trial_test_period = {'days': 0, 'hours': 0, 'minutes': 0}

        episode_params.update({'test_period': trial_test_period})

        # if target_period is None:
        #     target_period = {'days': 0, 'hours': 0, 'minutes': 0}

        trial_params['test_period'] = target_period

        # Setting target backshift:
        if use_target_backshift:
            trial_params['_test_period_backshift_delta'] =\
                datetime.timedelta(**trial_params['sample_duration']) - datetime.timedelta(**trial_test_period)

        episode_config = dict(
            class_ref=self.episode_class_ref,
            kwargs=dict(
                parsing_params=parsing_params,
                sampling_params=None,
                name='episode',
                task=task,
                log_level=log_level,
                _config_stack=None,
            ),
        )
        trial_config = dict(
            class_ref=self.trial_class_ref,
            kwargs=dict(
                parsing_params=parsing_params,
                sampling_params=episode_params,
                name='trial',
                task=task,
                frozen_time_split=frozen_time_split,
                log_level=log_level,
                _config_stack=[episode_config],
            ),
        )

        super(BTgymRandomDataDomain, self).__init__(
            dataframe=dataframe,
            parsing_params=parsing_params,
            sampling_params=trial_params,
            name=name,
            task=task,
            frozen_time_split=frozen_time_split,
            data_names=data_names,
            log_level=log_level,
            _config_stack=[episode_config, trial_config]
        )


class BTgymDataset2(BTgymRandomDataDomain):
    """
    Simple top-level data class, implements direct random episode sampling from data set induced by csv file,
    i.e it is a special case for `Trial=def=Episode`.
    """
    def __init__(
            self,
            dataframe,
            episode_duration=None,
            time_gap=None,
            start_00=False,
            start_weekdays=None,
            parsing_params=None,
            target_period=None,
            name='SimpleDataSet2',
            data_names=('default_asset',),
            log_level=WARNING,
            **kwargs
    ):
        """
        Args:
            dataframe:          pd.dataframe or iterable of pd.dataframes containing historic data;
            episode_duration:   dict, maximum episode duration in d:h:m, def={'days': 0, 'hours': 23, 'minutes': 55},
                                alias for `sample_duration`;
            time_gap:           dict, data time gap allowed within sample in d:h:m, def={'days': 0, 'hours': 6};
            start_00:           bool, episode start point will be shifted back to first record;
                                of the day (usually 00:00), def=False;
            start_weekdays:     list, only weekdays from the list will be used for sample start,
                                def=[0, 1, 2, 3, 4, 5, 6];
            target_period:      domain test(aka target) period. def={'days': 0, 'hours': 0, 'minutes': 0};
                                setting this param to non-zero duration forces data separation to train/test
                                subsets. Train data always precedes test one.
            parsing_params:     csv parsing options, see base class description for details;
            name:               str, instance name;
            log_level:          int, logbook.level;
            **kwargs:
        """
        # Default sample time duration:
        if episode_duration is None:
            self._episode_duration = dict(
                    days=0,
                    hours=23,
                    minutes=55,
                )
        else:
            self._episode_duration = episode_duration

        # Default data time gap allowed within sample:
        if time_gap is None:
            self._time_gap = dict(
                days=0,
                hours=6,
            )
        else:
            self._time_gap = time_gap

        # Default weekdays:
        if start_weekdays is None:
            start_weekdays = [0, 1, 2, 3, 4, 5, 6]

        trial_params = dict(
            sample_duration=self._episode_duration,
            start_weekdays=start_weekdays,
            start_00=start_00,
            time_gap=self._time_gap,
            # test_period={'days': 0, 'hours': 0, 'minutes': 0},
            test_period=target_period,
            expanding=False
        )
        episode_params = trial_params.copy()
        super(BTgymDataset2, self).__init__(
            dataframe=dataframe,
            parsing_params=parsing_params,
            trial_params=trial_params,
            episode_params=episode_params,
            target_period=target_period,
            name=name,
            data_names=data_names,
            log_level=log_level,
        )

