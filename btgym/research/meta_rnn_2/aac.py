
from btgym.research.gps.aac import GuidedAAC
from btgym.research.meta_rnn_2.env_runner import MetaEnvRunnerFn


class MetaAAC_0_0(GuidedAAC):
    """
    Meta-learning with RNN and GPS
    """
    def __init__(
            self,
            runner_fn_ref=MetaEnvRunnerFn,
            aac_lambda=1.0,
            guided_lambda=1.0,
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            _log_name='MetaA3C_0.0',
            **kwargs
    ):
        """

        Args:
            runner_fn_ref:
            _aux_render_modes:
            _log_name:
            **kwargs:
        """
        try:
            super(MetaAAC_0_0, self).__init__(
                runner_fn_ref=runner_fn_ref,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                _aux_render_modes=_aux_render_modes,
                _log_name=_log_name,
                **kwargs
            )
        except:
            msg = 'MetaAAC_0_0.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_sample_config(self, _new_trial=False):
        """
        Returns environment configuration parameters for next episode to sample.
        By default is simple stateful iterator,
        works correctly with `DTGymDataset` data class, repeating cycle:
            - sample `num_train_episodes` from source domain data,
            - sample `num_test_episodes` from target domain test data.

        Convention: supposed to override dummy method of local policy instance, see inside ._make_policy() method

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """
        # sess = tf.get_default_session()
        if self.current_train_episode < self.num_train_episodes:
            episode_type = 0  # train
            self.current_train_episode += 1
            self.log.debug(
                'c_1, c_train={}, c_test={}, type={}'.
                format(self.current_train_episode, self.current_test_episode, episode_type)
            )
        else:
            if self.current_test_episode < self.num_test_episodes:
                episode_type = 1  # test
                _new_trial = 1  # get test episode from target domain
                self.current_test_episode += 1
                self.log.debug(
                    'c_2, c_train={}, c_test={}, type={}'.
                    format(self.current_train_episode, self.current_test_episode, episode_type)
                )
            else:
                # cycle end, reset and start new (rec. depth 1)
                self.current_train_episode = 0
                self.current_test_episode = 0
                self.log.debug(
                    'c_3, c_train={}, c_test={}'.
                    format(self.current_train_episode, self.current_test_episode)
                )
                return self.get_sample_config(_new_trial=True)

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=_new_trial,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config