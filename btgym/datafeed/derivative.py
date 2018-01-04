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
import datetime

from .base import BTgymBaseData


class BTgymEpisode(BTgymBaseData):
    """
    End-user data class.
    `Episode` object contains single episode data sequence.
    Doesnt allows further sampling and data loading.
    Supposed to be converted to bt.datafeed object.
    """
    base_params = dict(
        filename=None,
        sample_class_ref=None,
        start_weekdays=[],
        start_00=False,
        sample_duration=dict(
            days=0,
            hours=0,
            minutes=0
        ),
        time_gap=dict(
            days=0,
            hours=0,
        ),
        test_period=dict(
            days=0,
            hours=0,
            minutes=0
        ),
        sample_expanding=None,
        sample_id=''
    )

    def __init__(self, **kwargs):
        #self.base_params.update(kwargs)
        super(BTgymEpisode, self).__init__(**self.base_params)
        assert datetime.timedelta(**self.test_period).total_seconds() == 0,\
            'Episode object doesnt support subset split, got: test_period={}'.format(self.test_period)

    def reset(self, **kwargs):
        raise RuntimeError('Episode object doesnt support .reset() method.')

    def sample(self, **kwargs):
        raise RuntimeError('Episode object doesnt support .sample() method.')


class BTgymDataTrial(BTgymBaseData):
    episode_params = dict(
        sample_class_ref=BTgymEpisode,
        start_weekdays=[0, 1, 2, 3, 4],
        start_00=False,
        sample_duration=dict(
            days=1,
            hours=23,
            minutes=55
        ),
        time_gap=dict(
            days=0,
            hours=5,
            minutes=0
        ),
        test_period=dict(
            days=0,
            hours=0,
            minutes=0
        ),
        sample_expanding=None,
        sample_name='episode_',
        metadata=dict(
            sample_num=0,
            type=0,
            trial_num=0,
        )
    )

    def __init__(self, episode_params=None, **kwargs):
        if episode_params is not None:
            self.episode_params.update(episode_params)
        self.episode_params.update(kwargs)
        super(BTgymDataTrial, self).__init__(**self.episode_params)


class BTgymDataDomain(BTgymBaseData):
    domain_params = dict(
        test_period=dict(
            days=0,
            hours=0,
            minutes=0
        )
    )
    trial_params = dict(
        sample_class_ref=BTgymDataTrial,
        start_weekdays=[0, 1, 2, 3, 4, 5, 6],
        start_00=False,
        sample_duration=dict(
            days=29,
            hours=23,
            minutes=55
        ),
        time_gap=dict(
            days=15,
            hours=0,
            minutes=0
        ),
        test_period=dict(
            days=0,
            hours=0,
            minutes=0
        ),
        sample_expanding=None,
        sample_name='trial_',
    )
    episode_params = dict(
        sample_class_ref=BTgymEpisode,
        start_weekdays=[0, 1, 2, 3],
        start_00=False,
        sample_duration=dict(
            days=1,
            hours=23,
            minutes=55
        ),
        time_gap=dict(
            days=5,
            hours=0,
            minutes=0
        ),
        test_period=dict(
            days=0,
            hours=0,
            minutes=0
        ),
        sample_expanding=None,
        sample_name='episode_',
        metadata=dict(
            sample_num=0,
            type=0,
            trial_num=0,
        )
    )

    def __init__(self, trial_params=None, episode_params=None, **kwargs):
        if trial_params is not None:
            self.trial_params.update(trial_params)

        self.trial_params.update(kwargs)
        # hacky:
        sample_test_period = self.trial_params['test_period']
        self.trial_params.update(self.domain_params)

        if episode_params is not None:
            self.episode_params.update(episode_params)

        super(BTgymDataDomain, self).__init__(**self.trial_params)
        self.sample_params.update(self.episode_params)
        self.sample_params.update({'test_period': sample_test_period})


class BTgymSimpleData(BTgymDataDomain):

    def __init__(self, **kwargs):
        # Depricated:
        if 'episode_duration' in kwargs.keys():
            kwargs['sample_duration'] = kwargs['episode_duration']

        super(BTgymSimpleData, self).__init__(trial_params=kwargs, episode_params=kwargs, **kwargs)


