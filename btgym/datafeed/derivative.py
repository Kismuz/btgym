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
            filename=None,
            parsing_params=None,
            sampling_params=None,
            name=None,
            task=0,
            log_level=WARNING,
            _config_stack=None,
    ):

        super(BTgymEpisode, self).__init__(
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=None,
            name='episode',
            task=task,
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
    Supports data train/test split.
    Do not use directly.
    """
    trial_params = dict(
        nested_class_ref=BTgymEpisode,
    )

    def __init__(
            self,
            filename=None,
            parsing_params=None,
            sampling_params=None,
            name=None,
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
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=sampling_params,
            name='Trial',
            task=task,
            log_level=log_level,
            _config_stack=_config_stack
        )


class BTgymRandomDataDomain(BTgymBaseData):
    """
    Top-level data class. Implements pipe::

        Domain.sample() --> Trial.sample() --> Episode.to_btfeed() --> bt.Startegy

    This particular class randomly samples Trials from provided dataset.

    Note:
        source/target domain split is not implemented yet.
    """
    def __init__(
            self,
            filename=None,
            parsing_params=None,
            trial_params=None,
            episode_params=None,
            name='RndDataDomain',
            task=0,
            log_level=WARNING,
    ):
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
        if parsing_params is None:
            parsing_params = dict(
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
            )

        try:
            # Hacky cause we want test period to be attr of Trial instance:
            trial_test_period = trial_params.pop('test_period')

        except(AttributeError, KeyError):
            trial_test_period = {'days': 0, 'hours': 0, 'minutes': 0}

        episode_params.update({'test_period': trial_test_period})

        episode_config = dict(
            class_ref=BTgymEpisode,
            kwargs=dict(
                parsing_params=parsing_params,
                sampling_params=None,
                name='Episode',
                task=task,
                log_level=log_level,
                _config_stack=None,
            ),
        )
        trial_config = dict(
            class_ref=BTgymDataTrial,
            kwargs=dict(
                parsing_params=parsing_params,
                sampling_params=episode_params,
                name='trial',
                task=task,
                log_level=log_level,
                _config_stack=[episode_config],
            ),
        )

        super(BTgymRandomDataDomain, self).__init__(
            filename=filename,
            parsing_params=parsing_params,
            sampling_params=trial_params,
            name=name,
            task=task,
            log_level=log_level,
            _config_stack=[episode_config, trial_config]
        )


class BTgymDataset(BTgymRandomDataDomain):
    """
    Simple top-level data class, implements direct random episode sampling from data set induced by csv file,
    i.e it is a special case for `Trial=def=DataDomain`.
    Doesnt support train/test split. Mainly for demo and debug purposes.
    """
    params_deprecated=dict(
        episode_len_days=('episode_duration', 'days'),
        episode_len_hours=('episode_duration','hours'),
        episode_len_minutes=('episode_duration', 'minutes'),
        time_gap_days=('time_gap', 'days'),
        time_gap_hours=('time_gap', 'hours')
    )

    def __init__(
            self,
            filename=None,
            episode_duration=None,
            time_gap=None,
            start_00=False,
            start_weekdays=None,
            parsing_params=None,
            name='SimpleDataSet',
            log_level=WARNING,
            **kwargs
    ):
        """
        Args:
            filename:           Str or list of str, file_names containing CSV historic data;
            episode_duration:   dict, maximum episode duration in d:h:m, def={'days': 0, 'hours': 23, 'minutes': 55},
                                alias for `sample_duration`;
            time_gap:           dict, data time gap allowed within sample in d:h:m, def={'days': 0, 'hours': 6};
            start_00:           bool, episode start point will be shifted back to first record;
                                of the day (usually 00:00), def=False;
            start_weekdays:     list, only weekdays from the list will be used for sample start,
                                def=[0, 1, 2, 3, 4, 5, 6];
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

        # Insert deprecated params, if any:
        for key, value in kwargs.items():
            if key in self.params_deprecated.keys():
                self.log.warning(
                    'Key: <{}> is deprecated, use: <{}> instead'.format(key, self.params_deprecated[key])
                )
                key1, key2 = self.params_deprecated[key]
                attr = getattr(self, key1)
                attr[key2] = value

        sampling_params=dict(
            sample_duration=self._episode_duration,
            start_weekdays=start_weekdays,
            start_00=start_00,
            time_gap=self._time_gap,
            test_period={'days': 0, 'hours': 0, 'minutes': 0},
            expanding=False
        )
        super(BTgymDataset, self).__init__(
            filename=filename,
            parsing_params=parsing_params,
            trial_params=sampling_params,
            episode_params=sampling_params,
            name=name,
            log_level=log_level,
        )


