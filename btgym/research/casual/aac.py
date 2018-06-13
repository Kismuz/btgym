import tensorflow as tf
import numpy as np
import time
import datetime

from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class CA3C(GuidedAAC):
    """
    Temporally dependant vanilla A3C. This is mot a meta-learning class.
    Requires stateful temporal data stream provider class such as btgym.datafeed.time.BTgymCasualDataDomain
    """

    def __init__(
            self,
            runner_config=None,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            test_slowdown_steps=1,
            episode_sample_params=(1.0, 1.0),
            trial_sample_params=(1.0, 1.0),
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            _use_target_policy=False,
            name='CasualA3C',
            **kwargs
    ):
        try:
            if runner_config is None:
                self.runner_config = {
                    'class_ref': BaseSynchroRunner,
                    'kwargs': {
                        'data_sample_config': {'mode': 0},
                        'test_conditions': {
                            'state': {
                                'metadata': {
                                    'trial_type': 1,  # only test episode from target dom. considered test one
                                    'type': 1
                                }
                            }
                        },
                        'slowdown_steps': test_slowdown_steps,
                        'name': '',
                    },
                }
            else:
                self.runner_config = runner_config

            # Trials sampling control:
            self.num_source_trials = trial_source_target_cycle[0]
            self.num_target_trials = trial_source_target_cycle[-1]
            self.num_episodes_per_trial = num_episodes_per_trial

            self.test_slowdown_steps = test_slowdown_steps

            self.episode_sample_params = episode_sample_params
            self.trial_sample_params = trial_sample_params

            self.global_timestamp = 0

            self.current_source_trial = 0
            self.current_target_trial = 0
            self.current_trial_mode = 0  # source
            self.current_episode = 1

            super(CA3C, self).__init__(
                runner_config=self.runner_config,
                _aux_render_modes=_aux_render_modes,
                name=name,
                **kwargs
            )
        except:
            msg = '{}.__init()__ exception occurred'.format(name) + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_sample_config(self, **kwargs):
        """
        Returns environment configuration parameters for next episode to sample.

        Here we always prescribe to sample test episode from source or target domain.

        Args:
              kwargs:     not used

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """

        new_trial = 0
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

        if self.task == 0:
            trial_sample_type = 1

        else:
            trial_sample_type = self.current_trial_mode

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=1,
                timestamp= self.global_timestamp,
                b_alpha=self.episode_sample_params[0],
                b_beta=self.episode_sample_params[-1]
            ),
            trial_config=dict(
                get_new=new_trial,
                sample_type=trial_sample_type,
                timestamp=self.global_timestamp,
                b_alpha=self.trial_sample_params[0],
                b_beta=self.trial_sample_params[-1]
            )
        )
        return sample_config

    def process(self, sess, **kwargs):
        try:
            sess.run(self.sync_pi)
            # Get data configuration:
            data_config = self.get_sample_config()

            # self.log.warning('data_config: {}'.format(data_config))

            # If this step data comes from source or target domain
            is_test = data_config['trial_config']['sample_type'] and data_config['episode_config']['sample_type']

            # self.log.warning('is_test: {}'.format(is_test))

            if is_test:
                if self.task == 0:
                    self.process_test(sess, data_config)

                else:
                    pass

            else:
                self.process_train(sess, data_config)

        except:
            msg = 'process() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def process_test(self, sess, data_config):
        data = {}
        done = False
        # Set target episode beginning to be at current timepoint:
        data_config['trial_config']['align_left'] = 1
        self.log.info('test episode started...')

        while not done:
            #sess.run(self.sync_pi)

            data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config,
                policy_sync_op=self.sync_pi,  # update policy after each single step instead of single rollout
            )
            done = np.asarray(data['terminal']).any()

            # self.log.warning('test episode done: {}'.format(done))

            self.global_timestamp = data['on_policy'][0]['state']['metadata']['timestamp'][-1]

            # # Wait for other workers to catch up with training:
            # start_global_step = sess.run(self.global_step)
            # while self.test_skeep_steps >= sess.run(self.global_step) - start_global_step:
            #     time.sleep(self.test_sleep_time)

            self.log.info(
                'test episode rollout done, global_time: {}, global_step: {}'.format(
                    datetime.datetime.fromtimestamp(self.global_timestamp),
                    sess.run(self.global_step)
                )
            )

        self.log.notice(
            'test episode finished, global_time set: {}'.format(
                datetime.datetime.fromtimestamp(self.global_timestamp)
            )
        )
        self.log.notice(
            'final value: {:8.2f} after {} steps @ {}'.format(
                data['on_policy'][0]['info']['broker_value'][-1],
                data['on_policy'][0]['info']['step'][-1],
                data['on_policy'][0]['info']['time'][-1],
            )
        )
        data['ep_summary'] = [None]  # We only test here, want no train NAN's
        self.process_summary(sess, data)

    def process_train(self, sess, data_config):
        data = {}
        done = False
        # Set source episode to be sampled uniformly from test interval:
        data_config['trial_config']['align_left'] = 0
        # self.log.warning('train episode started...')

        while not done:
            sess.run(self.sync_pi)

            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config
            )
            done = np.asarray(data['terminal']).any()
            feed_dict = self.process_data(sess, data, is_train=True, pi=self.local_network)

            if wirte_model_summary:
                fetches = [self.train_op, self.model_summary_op, self.inc_step]
            else:
                fetches = [self.train_op, self.inc_step]

            fetched = sess.run(fetches, feed_dict=feed_dict)

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            self.process_summary(sess, data, model_summary)

            self.local_steps += 1

        # self.log.warning(
        #     'train episode finished at {} vs was_global_time: {}'.format(
        #         data['on_policy'][0]['info']['time'][-1],
        #         datetime.datetime.fromtimestamp(data['on_policy'][0]['state']['metadata']['timestamp'][-1])
        #
        #     )
        # )


class CA3Ca(CA3C):
    """
    + Adaptive iteratations.
    """

    def __init__(self, name='CasualAdaA3C', **kwargs):
        super(CA3Ca, self).__init__(name=name, **kwargs)

    def _make_loss(self, **kwargs):
        aac_loss, summaries = self._make_base_loss(**kwargs)

        # Guidance annealing:
        if self.guided_decay_steps is not None:
            self.guided_lambda_decayed = tf.train.polynomial_decay(
                self.guided_lambda,
                self.global_step + 1,
                self.guided_decay_steps,
                0,
                power=1,
                cycle=False,
            )
        else:
            self.guided_lambda_decayed = self.guided_lambda
        # Switch to zero when testing - prevents information leakage:
        self.train_guided_lambda = self.guided_lambda_decayed * tf.cast(self.local_network.train_phase, tf.float32)

        self.guided_loss, guided_summary = self.expert_loss(
            pi_actions=self.local_network.on_logits,
            expert_actions=self.local_network.expert_actions,
            name='on_policy',
            verbose=True,
            guided_lambda=self.train_guided_lambda
        )
        loss = self.aac_lambda * aac_loss + self.guided_loss

        summaries += guided_summary

        self.log.notice(
            'guided_lambda: {:1.6f}, guided_decay_steps: {}'.format(self.guided_lambda, self.guided_decay_steps)
        )

        if hasattr(self.local_network, 'meta'):
            self.log.notice('meta_policy enabled')
            summaries += self.local_network.meta.summaries

        return loss, summaries

    def _make_train_op(self, pi, pi_prime, pi_global):
        """
        Defines training op graph and supplementary sync operations.

        Args:
            pi:                 policy network obj.
            pi_prime:           optional policy network obj.
            pi_global:          shared policy network obj. hosted by parameter server

        Returns:
            tensor holding training op graph;
        """

        # Each worker gets a different set of adam optimizer parameters:
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)

        # Clipped gradients:
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, pi.var_list),
            40.0
        )
        self.grads_global_norm = tf.global_norm(self.grads)
        # Copy weights from the parameter server to the local model
        self.sync = self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
        )
        if self.use_target_policy:
            # Copy weights from new policy model to target one:
            self.sync_pi_prime = tf.group(
                *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
            )
        grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(pi.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in['external'])[0])

        self.local_network.meta.grads_and_vars = list(
            zip(self.local_network.meta.grads, self.network.meta.var_list)
        )
        self.meta_opt = tf.train.GradientDescentOptimizer(self.local_network.meta.learn_rate)

        self.meta_train_op = self.meta_opt.apply_gradients(self.local_network.meta.grads_and_vars)

        self.local_network.meta.sync_slot_op = tf.assign(
            self.local_network.meta.cluster_averages_slot,
            self.network.meta.cluster_averages_slot,
        )

        self.local_network.meta.send_stat_op = tf.scatter_nd_update(
            self.network.meta.cluster_averages_slot,
            [[0, self.task], [1, self.task]],
            [
                self.local_network.meta.cluster_averages_slot[0, self.task],
                self.local_network.meta.cluster_averages_slot[1, self.task]
            ]
        )
        self.local_network.meta.global_reset = self.network.meta.reset

        train_op = self.optimizer.apply_gradients(grads_and_vars)

        self.local_network.meta.ppp = self.network.meta

        self.log.debug('train_op defined')
        return tf.group([train_op, self.meta_train_op])

    def process_test(self, sess, data_config):
        data = {}
        done = False
        # Set target episode beginning to be at current timepoint:
        data_config['trial_config']['align_left'] = 1

        self.log.info('test episode started...')

        while not done:

            # TODO: temporal, refract
            # self.local_network.meta.global_reset()
            sess.run(self.sync_pi)

            data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config,
                rollout_length=2,
                policy_sync_op=self.sync_pi,  # update policy after each single step instead of single rollout
            )
            done = np.asarray(data['terminal']).any()

            # self.log.warning('test episode done: {}'.format(done))

            self.global_timestamp = data['on_policy'][0]['state']['metadata']['timestamp'][-1]

            # # Wait for other workers to catch up with training:
            # start_global_step = sess.run(self.global_step)
            # while self.test_skeep_steps >= sess.run(self.global_step) - start_global_step:
            #     time.sleep(self.test_sleep_time)

            self.log.info(
                'test episode rollout done, global_time: {}, global_step: {}'.format(
                    datetime.datetime.fromtimestamp(self.global_timestamp),
                    sess.run(self.global_step)
                )
            )
            # Now can train on already past data:
            feed_dict = self.process_data(sess, data, is_train=True, pi=self.local_network)

            fetches = [self.train_op, self.model_summary_op, self.inc_step]

            fetched = sess.run(fetches, feed_dict=feed_dict)

            model_summary = fetched[-2]

            data['ep_summary'] = [None]  # We only test here, want no train NAN's
            self.process_summary(sess, data, model_summary)

        self.log.notice(
            'test episode finished, global_time set: {}'.format(
                datetime.datetime.fromtimestamp(self.global_timestamp)
            )
        )
        self.log.notice(
            'final value: {:8.2f} after {} steps @ {}'.format(
                data['on_policy'][0]['info']['broker_value'][-1],
                data['on_policy'][0]['info']['step'][-1],
                data['on_policy'][0]['info']['time'][-1],
            )
        )

    def process_train(self, sess, data_config):
        data = {}
        done = False
        # Set source episode to be sampled uniformly from test interval:
        data_config['trial_config']['align_left'] = 0
        # self.log.warning('train episode started...')

        while not done:
            sess.run(self.sync_pi)

            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config,
                policy_sync_op=None
            )
            done = np.asarray(data['terminal']).any()
            feed_dict = self.process_data(sess, data, is_train=True, pi=self.local_network)

            if wirte_model_summary:
                fetches = [self.local_network.on_vf, self.train_op, self.model_summary_op, self.inc_step]
            else:
                fetches = [self.local_network.on_vf, self.train_op, self.inc_step]

            sess.run(self.local_network.meta.sync_slot_op)

            fetched = sess.run(fetches, feed_dict=feed_dict)

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            self.process_summary(sess, data, model_summary)

            # Meta:
            train_stat = fetched[0] - 10

            if sess.run(self.network.meta.cluster_averages_slot)[0, self.task] == 0:
                self.local_network.meta.reset()

            self.local_network.meta.update(train_stat)

            sess.run(self.local_network.meta.send_stat_op)

            self.local_steps += 1


