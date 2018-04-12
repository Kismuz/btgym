import tensorflow as tf
import numpy as np

from btgym.research.gps.aac import GuidedAAC
from btgym.research.metalearn_2._env_runner import MetaEnvRunnerFn
from btgym.algorithms.runner import RunnerThread
from btgym.algorithms.memory import Memory
from btgym.algorithms.nn.losses import ppo_loss_def


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
                name=_log_name,
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


class MetaAAC_0_1(GuidedAAC):
    """
    Asynchronous implementation of `gradient-based meta-learning algorithm suitable
    for adaptation in dynamically changing` environments, as from paper:
        Maruan Al-Shedivat et al.,
        "Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments"
        https://arxiv.org/abs/1710.03641

         Da Li et al.,
         "Learning to Generalize: Meta-Learning for Domain Generalization"
         https://arxiv.org/abs/1710.03463

    """

    def __init__(
            self,
            runner_fn_ref=MetaEnvRunnerFn,
            aac_lambda=1.0,
            guided_lambda=1.0,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            _log_name='MetaA3C_0.1',
            **kwargs
    ):
        try:
            super(MetaAAC_0_1, self).__init__(
                #on_policy_loss=ppo_loss_def,
                #off_policy_loss=ppo_loss_def,
                runner_fn_ref=runner_fn_ref,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                _use_target_policy=True,
                _aux_render_modes=_aux_render_modes,
                name=_log_name,
                **kwargs
            )

            # Trials sampling control:
            self.num_source_trials = trial_source_target_cycle[0]
            self.num_target_trials = trial_source_target_cycle[-1]
            self.num_episodes_per_trial = num_episodes_per_trial

            # Note that only master (test runner) is requesting trials

            self.current_source_trial = 0
            self.current_target_trial = 0
            self.current_trial_mode = 0  # source
            self.current_episode = 0

        except:
            msg = 'MetaAAC_0_1.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def worker_device_callback_0(self):
        """Behavioural policies for test and train data."""
        # Make one new
        self.mu = self._make_policy('local_mu')
        # Just rename the other:
        self.mu_prime = self.local_network_prime

    def _make_train_op(self):
        """
        Overrides base method.
        Defines gradients cross-update rule.

        Returns:
            tensor holding training op graph;
        """
        # Clipped gradients:
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, self.local_network.var_list),
            40.0
        )
        # gradients wrt parameters:
        test_grads_and_vars = list(zip(self.grads, self.network_prime.var_list))
        train_grads_and_vars = list(zip(self.grads, self.network.var_list))

        # Copy weights from the parameter server to the local behavioural `train` policy:
        self.sync_mu_global = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.mu.var_list, self.network.var_list)]
        )
        # Copy weights from the parameter server to the local behavioural `test` policy:
        self.sync_mu_prime_global = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.mu_prime.var_list, self.network_prime.var_list)]
        )
        # Copy from local behavioural to trainable:
        self.sync_pi_mu = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.mu.var_list)]
        )
        self.sync_pi_mu_prime = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.mu_prime.var_list)]
        )

        self.sync = [self.sync_mu_prime_global, self.sync_mu_global]

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.local_network.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(self.local_network.on_state_in['external'])[0])

        #self.optimizer = {
        #    'train': tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5),
        #    'test': tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5),
        #}
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)

        # self.optimizer = tf.train.RMSPropOptimizer(
        #    learning_rate=train_learn_rate,
        #    decay=self.opt_decay,
        #    momentum=self.opt_momentum,
        #    epsilon=self.opt_epsilon,
        # )

        # Cross-over gradient update:
        train_on_train_op = self.optimizer.apply_gradients(test_grads_and_vars)
        train_on_test_op = self.optimizer.apply_gradients(train_grads_and_vars)

        self.log.debug('train/test_op defined')
        return {'train': train_on_train_op, 'test': train_on_test_op}

    def _make_runners(self):
        """
        Overrides base method.
        Expects receive list of two master/salve environments. Assigns train policy to Master environment ,
        test - to salve.

        Returns:
            list of runners
        """
        # Replay memory_config:
        if self.use_memory:
            memory_config = dict(
                class_ref=Memory,
                kwargs=dict(
                    history_size=self.replay_memory_size,
                    max_sample_size=self.replay_rollout_length,
                    priority_sample_size=self.rp_sequence_size,
                    reward_threshold=self.rp_reward_threshold,
                    use_priority_sampling=self.use_reward_prediction,
                    task=self.task,
                    log_level=self.log_level,
                )
            )
        else:
            memory_config = None

        assert len(self.env_list) == 2, 'Expected pair of environments, got: {}'.format(self.env_list)

        # Make runners, assign different policies:
        runners = []
        # Now we get master environment [first in pair] provide test data and slave environment  - train data.
        # Use mu-policies for acting.
        for env, policy, task in zip(
                self.env_list,
                [self.mu_prime, self.mu],
                ['test', 'train']
        ):
            runners.append(
                RunnerThread(
                    env=env,
                    policy=policy,
                    runner_fn_ref=self.runner_fn_ref,
                    task='{}_{}'.format(self.task, task),
                    rollout_length=self.rollout_length,  # ~20
                    episode_summary_freq=self.episode_summary_freq,
                    env_render_freq=self.env_render_freq,
                    test=self.test_mode,
                    ep_summary=self.ep_summary,
                    memory_config=memory_config,
                    log_level=self.log_level,
                )
            )
        self.log.debug('thread-runners ok.')
        return runners

    def get_sample_config(self, mode=0):
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
            new_trial = 1  # slave env. gets new trial anyway

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=mode,
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

    def get_data(self):
        """
        Collect rollouts from every environment.
        Overrides base method, does not merges data_streams.

        Returns:
            dictionary of lists of data streams collected from every runner
        """
        data_streams = [get_it() for get_it in self.data_getter]

        wrapped_streams = [
            {key: [data_streams[0][key]] for key in data_streams[0].keys()},
            {key: [data_streams[-1][key]] for key in data_streams[-1].keys()},
        ]

        return wrapped_streams

    def process(self, sess):
        """

        Args:
            sess (tensorflow.Session):   tf session obj.
        """
        # Quick wrap to get direct traceback from this trainer if something goes wrong:
        try:
            # Copy mu and mu_prime from global:
            sess.run(self.sync)

            # Collect data from child thread runners:
            test_data, train_data = self.get_data()
            # TODO: data checks!!!

            # Source or target:
            try:
                is_source = not np.asarray(
                    [env['state']['metadata']['trial_type'] for env in test_data['on_policy']]
                ).any()
            except KeyError:
                is_source = True

            if is_source:
                # If there is no any test rollouts:

                # Say `No` to redundant summaries:
                wirte_model_summary =\
                    self.local_steps % self.model_summary_freq == 0

                # Do a train step on train data:
                # here we send gradients to global_prime network
                sess.run(self.sync_pi_mu)
                train_feed_dict = self.process_data(sess,,,,,,,,
                fetches = [self.train_op['train']]
                fetched = sess.run(fetches, feed_dict=train_feed_dict)  # TODO: use train summaries as well

                # Do a train step on test data:
                # grads sent to global network
                sess.run(self.sync_pi_mu_prime)
                test_feed_dict = self.process_data(sess,,,,,,,,
                fetches = [self.train_op['test']]

                if wirte_model_summary:
                    fetches = fetches + [self.model_summary_op, self.inc_step]
                else:
                    fetches = fetches + [self.inc_step]

                fetched = sess.run(fetches, feed_dict=test_feed_dict)

                if wirte_model_summary:
                    model_summary = fetched[-2]

                else:
                    model_summary = None

                self.local_steps += 1  # only update on train steps

            else:
                model_summary = None

            # Write down summaries (use test data):
            self.process_summary(sess, test_data, model_summary)

            # print debug info:
            #for k, v in fetched[1].items():
            #    print('{}: {}'.format(k,v))
            #print('\n')

            #for k, v in feed_dict.items():
            #    try:
            #        print(k, v.shape)
            #    except:
            #        print(k, type(v))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        except:
            msg = 'process() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)