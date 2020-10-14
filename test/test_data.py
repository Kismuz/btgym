import unittest

from btgym.datafeed.derivative import BTgymDataset, BTgymRandomDataDomain
from btgym.datafeed.stateful import BTgymSequentialDataDomain

filename = '../examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

trial_params = dict(
    start_weekdays={0, 1, 2, 3, 4, 5, 6},
    sample_duration={'days': 8, 'hours': 0, 'minutes': 0},
    start_00=False,
    time_gap={'days': 4, 'hours': 0},
    test_period={'days': 2, 'hours': 0, 'minutes': 0},
)

episode_params = dict(
    start_weekdays={0, 1, 2, 3, 4, 5, 6},
    sample_duration={'days': 0, 'hours': 23, 'minutes': 55},
    start_00=False,
    time_gap={'days': 0, 'hours': 10},
)

target_period = {'days': 29, 'hours': 0, 'minutes': 0}

log_level = 12


class DomainTest(unittest.TestCase):
    """Testing domain class"""

    def test_domain_target_period_bounds_fail(self):
        """
        Tests non-zero train size check
        """
        with self.assertRaises(AssertionError) as cm:
            domain = BTgymDataset(
                filename=filename,
                episode_duration={'days': 0, 'hours': 23, 'minutes': 55},
                start_00=False,
                time_gap={'days': 0, 'hours': 10},
                target_period={'days': 360, 'hours': 0, 'minutes': 0}
            )
            domain.reset()

    def test_BTgymDataset_sampling_bounds_consistency(self):
        """
        Any train trial mast precede any test period.
        Same true for train/test episodes.
        """
        domain = BTgymDataset(
            filename=filename,
            episode_duration={'days': 0, 'hours': 23, 'minutes': 55},
            start_00=False,
            time_gap={'days': 0, 'hours': 10},
            target_period={'days': 40, 'hours': 0, 'minutes': 0}
        )
        domain.reset()
        sup_train_time = 0
        inf_test_time = domain.data[-2:-1].index[0].value
        for i in range(100):
            train_trial = domain.sample(get_new=True, sample_type=0)
            last_train_time = train_trial.data[-2:-1].index[0].value
            if last_train_time > sup_train_time:
                sup_train_time = last_train_time

            test_trial = domain.sample(get_new=True, sample_type=1)
            first_test_time = test_trial.data[0:1].index[0].value
            if first_test_time < inf_test_time:
                inf_test_time = first_test_time

            with self.subTest(msg='sub_{}'.format(i), train_trial=train_trial.filename, test_trial=test_trial.filename):
                self.assertLess(sup_train_time, inf_test_time)
                with self.subTest('Train/test should be irrelevant dor Dataset episodes'):
                    train_trial.reset()
                    test_trial.reset()
                    episode_1 = test_trial.sample(get_new=True, sample_type=1)
                    episode_2 = train_trial.sample(get_new=True, sample_type=1)

    def test_BTgymRandomDataDomain_sampling_bounds_consistency(self):
        """
        Any train trial mast precede any test period.
        Same true for train/test episodes.
        """
        rnd_domain = BTgymRandomDataDomain(
            filename=filename,
            trial_params=trial_params,
            episode_params=episode_params,
            target_period={'days': 40, 'hours': 0, 'minutes': 0},
            log_level=log_level,
        )

        domains = [rnd_domain]

        for domain in domains:
            with self.subTest(domain=type(domain)):
                domain.reset()
                sup_train_time = 0
                inf_test_time = domain.data[-2:-1].index[0].value
                for i in range(50):
                    train_trial = domain.sample(get_new=True, sample_type=0)
                    last_train_time = train_trial.data[-2:-1].index[0].value
                    if last_train_time > sup_train_time:
                        sup_train_time = last_train_time

                    test_trial = domain.sample(get_new=True, sample_type=1)
                    first_test_time = test_trial.data[0:1].index[0].value
                    if first_test_time < inf_test_time:
                        inf_test_time = first_test_time

                    self.assertLess(sup_train_time, inf_test_time)

                    with self.subTest(train_trial=train_trial.filename, test_trial=test_trial.filename):
                        trials = [train_trial, test_trial]

                        for trial in trials:
                            with self.subTest(actual_trial=trial.filename):
                                trial.reset()
                                e_sup_time = 0
                                e_inf_time = trial.data[-2:-1].index[0].value
                                for i in range(20):
                                    train_episode = trial.sample(get_new=True, sample_type=0)
                                    e_last_time = train_episode.data[-2:-1].index[0].value
                                    if e_last_time > e_sup_time:
                                        e_sup_time = e_last_time

                                    test_episode = trial.sample(get_new=True, sample_type=1)
                                    e_first_time = test_episode.data[0:1].index[0].value
                                    if e_first_time < e_inf_time:
                                        e_inf_time = e_first_time

                                    with self.subTest(
                                            train_episode=train_episode.filename,
                                            test_episode=test_episode.filename
                                    ):
                                        self.assertLess(e_sup_time, e_inf_time)

    def _BTgymSequentialDataDomain_sampling_bounds_consistency(self):
        """
        Any train trial mast precede any test period.
        Same true for train/test episodes.
        """
        seq_domain = BTgymSequentialDataDomain(
            filename=filename,
            trial_params=trial_params,
            episode_params=episode_params,
            log_level=log_level,
        )
        domains = [seq_domain]

        for domain in domains:
            with self.subTest(domain=type(domain)):
                domain.reset()
                cardinality = domain.total_samples
                print('BTgymSequentialDataDomain cardinality:', cardinality)

                last_trial_sup = 0
                for i in range(cardinality - 1):
                    trial = domain.sample(get_new=True)
                    trial_sup = trial.data[-2:-1].index[0].value

                    self.assertLess(last_trial_sup, trial_sup)

                    with self.subTest(trial=trial.filename):
                        trial.reset()
                        e_train_sup_time = 0
                        e_test_inf_time = trial.data[-2:-1].index[0].value
                        for i in range(10):
                            train_episode = trial.sample(get_new=True, sample_type=0)
                            e_last_time = train_episode.data[-2:-1].index[0].value
                            if e_last_time > e_train_sup_time:
                                e_train_sup_time = e_last_time

                            test_episode = trial.sample(get_new=True, sample_type=1)
                            e_first_time = test_episode.data[0:1].index[0].value
                            if e_first_time < e_test_inf_time:
                                e_test_inf_time = e_first_time

                            with self.subTest(
                                    msg='Train episode precedes Test',
                                    train_episode=train_episode.filename,
                                    test_episode=test_episode.filename
                            ):
                                self.assertLess(e_train_sup_time, e_test_inf_time)

                            with self.subTest(
                                    msg='Test episodes form partition',
                                    test_episode=test_episode.filename
                            ):
                                self.assertLess(last_trial_sup, e_test_inf_time)


if __name__ == '__main__':
    unittest.main()
