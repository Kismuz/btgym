import tensorflow as tf
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class AMLDG_1(GuidedAAC):
    """
    Asynchronous implementation of MLDG algorithm (by Da Li et al.)
    for one-shot adaptation in dynamically changing environments.

    Defines meta-test task as one-roll-ahead of train one.
    Uses per-episode replay buffer as meta-train distribution support.

    Papers:
        Da Li et al.,
         "Learning to Generalize: Meta-Learning for Domain Generalization"
         https://arxiv.org/abs/1710.03463

        Maruan Al-Shedivat et al.,
        "Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments"
        https://arxiv.org/abs/1710.03641
    """

    def __init__(
            self,
            runner_config=None,
            fast_opt_learn_rate=1e-3,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='AMLDG1',
            **kwargs
    ):
        try:
            if runner_config is None:
                self.runner_config = {
                    'class_ref': BaseSynchroRunner,
                    'kwargs': {
                        'data_sample_config': {'mode': 1},
                        'name': '',
                    },
                }
            else:
                self.runner_config = runner_config

            # Trials sampling control:
            self.num_source_trials = trial_source_target_cycle[0]
            self.num_target_trials = trial_source_target_cycle[-1]
            self.num_episodes_per_trial = num_episodes_per_trial

            self.current_source_trial = 0
            self.current_target_trial = 0
            self.current_trial_mode = 0  # source
            self.current_episode = 0

            self.fast_opt_learn_rate = fast_opt_learn_rate

            super(AMLDG_1, self).__init__(
                runner_config=runner_config,
                _use_target_policy=True,
                _aux_render_modes=_aux_render_modes,
                name=name,
                **kwargs
            )
        except:
            msg = '{}.__init()__ exception occurred'.format(name) + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_train_op(self):
        """
        Defines training op graph and supplementary sync operations.

        Returns:
            tensor holding training op graph;
        """
        pi = self.local_network
        pi_prime = self.local_network_prime
        pi_global = self.network

        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)
        self.fast_optimizer = tf.train.GradientDescentOptimizer(self.fast_opt_learn_rate)

        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, self.local_network.var_list),
            40.0
        )
        # Copy weights from the parameter server to the local model
        self.sync = self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.network.var_list)]
        )
        if self.use_target_policy:
            # Copy weights from new policy model to target one:
            self.sync_pi_prime = tf.group(
                *[v1.assign(v2) for v1, v2 in zip(self.local_network_prime.var_list, self.local_network.var_list)]
            )
        grads_and_vars = list(zip(self.grads, self.network.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.local_network.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(self.local_network.on_state_in['external'])[0])

        train_op = self.optimizer.apply_gradients(grads_and_vars)
        self.log.debug('train_op defined')
        return train_op

    def get_sample_config(self, mode=1):
        """
        Returns environment configuration parameters for next episode to sample.

        Args:
              mode:     bool, False for slave (train data), True for master (test data)

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """

        new_trial = 0
        if mode:
            # Only master environment updates counters:
            if self.current_episode >= self.num_episodes_per_trial:
                # Reset episode counter:
                self.current_episode = 0

                # Request new trial:
                new_trial = 1
                # Decide on trial type (source/target):
                if self.current_source_trial >= self.num_source_trials:
                    # Time to switch to target mode:
                    self.current_trial_mode = 1
                    # Reset counters:
                    self.current_source_trial = 0
                    self.current_target_trial = 0

                if self.current_target_trial >= self.num_target_trials:
                    # Vise versa:
                    self.current_trial_mode = 0
                    self.current_source_trial = 0
                    self.current_target_trial = 0

                # Update counter:
                if self.current_trial_mode:
                    self.current_target_trial += 1
                else:
                    self.current_source_trial += 1

            self.current_episode += 1
        else:
            new_trial = 1

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=int(not mode),
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=new_trial,
                sample_type=self.current_trial_mode,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config

