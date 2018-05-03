
import unittest
import datetime
from .time import BTgymTimeDataDomain

from logbook import WARNING, INFO, DEBUG


filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

trial_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 10, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 5, 'hours': 0},
        test_period={'days': 2, 'hours': 0, 'minutes': 0},
    )
episode_params=dict(
    start_weekdays={0, 1, 2, 3, 4, 5, 6},
    sample_duration={'days': 0, 'hours': 23, 'minutes': 55},
    start_00=False,
    time_gap={'days': 0, 'hours': 10},
)


log_level = INFO


class TimeDomainTest(unittest.TestCase):
    """Testing time domain class"""

    def test_BTgymDataset_sampling_bounds_consistency(self):
        """
        Any train trial mast precede any test period.
        Same true for train/test episodes.
        """
        domain = BTgymTimeDataDomain(
            filename=filename,
            trial_params=trial_params,
            episode_params=episode_params,
            log_level=log_level,
        )
        domain.reset()
        timestamp = domain.global_timestamp

        for i in range(100):
            global_time = datetime.datetime.fromtimestamp(timestamp)
            print('\nGLOBAL_TIME: {}'.format(global_time))
            source_trial = domain.sample(get_new=True, sample_type=0, timestamp=timestamp)
            # print(source_trial.filename)
            target_trial = domain.sample(get_new=True, sample_type=1, timestamp=timestamp)
            # print(target_trial.filename)

            source_trial.reset()
            target_trial.reset()
            print('\nSource episodes:')
            ep_s_0 = source_trial.sample(get_new=True, sample_type=0, timestamp=timestamp)
            ep_s_1 = source_trial.sample(get_new=True, sample_type=1, timestamp=timestamp)
            print('\nTarget episodes:')
            ep_t_0 = target_trial.sample(get_new=True, sample_type=0, timestamp=timestamp)
            ep_t_1 = target_trial.sample(get_new=True, sample_type=1, timestamp=timestamp)

            self.assertLess(ep_s_0.data.index[-1], ep_s_1.data.index[0])
            self.assertLessEqual(ep_s_1.data.index[-1], global_time)
            self.assertLess(ep_t_0.data.index[-1], ep_t_1.data.index[0])
            self.assertLessEqual(global_time, ep_t_1.data.index[0])
            timestamp += 10000



if __name__ == '__main__':
    unittest.main()
