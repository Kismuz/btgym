import tensorflow as tf
import numpy as np
import time
import datetime

from btgym.algorithms import BaseAAC
from btgym.algorithms.math_utils import cat_entropy
# from btgym.algorithms.runner.synchro import BaseSynchroRunner
from btgym.research.encoder_test.runner import RegressionRunner


# class EncoderClassifier(BaseAAC):
#     """
#     `Fake AAC` class meant to test policy state encoder ability to predict price movement
#     as an isolated classification/regression problem.
#     """
#
#     def __init__(
#             self,
#             runner_config=None,
#             trial_source_target_cycle=(1, 0),
#             num_episodes_per_trial=1,  # one-shot adaptation
#             test_slowdown_steps=0,
#             episode_sample_params=(1.0, 1.0),
#             trial_sample_params=(1.0, 1.0),
#             aac_lambda=0,
#             class_lambda=1.0,
#             class_use_rnn=True,
#             _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
#             _use_target_policy=False,
#             name='EncoderClassifier',
#             **kwargs
#     ):
#         try:
#             if runner_config is None:
#                 self.runner_config = {
#                     'class_ref': BaseSynchroRunner,
#                     'kwargs': {
#                         'data_sample_config': {'mode': 0},
#                         'test_conditions': {
#                             'state': {
#                                 'metadata': {
#                                     'trial_type': 1,  # only test episode from target dom. considered test one
#                                     'type': 1
#                                 }
#                             }
#                         },
#                         'slowdown_steps': test_slowdown_steps,
#                         'name': '',
#                     },
#                 }
#             else:
#                 self.runner_config = runner_config
#
#             # Trials sampling control:
#             self.num_source_trials = trial_source_target_cycle[0]
#             self.num_target_trials = trial_source_target_cycle[-1]
#             self.num_episodes_per_trial = num_episodes_per_trial
#
#             self.aac_lambda = aac_lambda
#             self.class_lambda = class_lambda
#             self.class_use_rnn = class_use_rnn
#
#             self.test_slowdown_steps = test_slowdown_steps
#
#             self.episode_sample_params = episode_sample_params
#             self.trial_sample_params = trial_sample_params
#
#             self.global_timestamp = 0
#
#             self.current_source_trial = 0
#             self.current_target_trial = 0
#             self.current_trial_mode = 0  # source
#             self.current_episode = 1
#
#             super(EncoderClassifier, self).__init__(
#                 runner_config=self.runner_config,
#                 aux_render_modes=_aux_render_modes,
#                 name=name,
#                 **kwargs
#             )
#         except:
#             msg = '{}.__init()__ exception occurred'.format(name) + \
#                   '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
#             self.log.exception(msg)
#             raise RuntimeError(msg)
#
#     def _make_loss(self, pi, pi_prime, name='base', verbose=True, **kwargs):
#         """
#         Defines policy state encoder classification loss, placeholders and summaries.
#
#         Args:
#             pi:                 policy network obj.
#             pi_prime:           optional policy network obj.
#             name:               str, name scope
#             verbose:            summary level
#
#         Returns:
#             tensor holding estimated loss graph
#             list of related summaries
#         """
#         with tf.name_scope(name):
#             # On-policy AAC loss definition:
#             pi.on_pi_act_target = tf.placeholder(
#                 tf.float32, [None, self.ref_env.action_space.n], name="on_policy_action_pl"
#             )
#             pi.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
#             pi.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")
#
#             clip_epsilon = tf.cast(self.clip_epsilon * self.learn_rate_decayed / self.opt_learn_rate, tf.float32)
#
#             on_pi_loss, on_pi_summaries = self.on_policy_loss(
#                 act_target=pi.on_pi_act_target,
#                 adv_target=pi.on_pi_adv_target,
#                 r_target=pi.on_pi_r_target,
#                 pi_logits=pi.on_logits,
#                 pi_vf=pi.on_vf,
#                 pi_prime_logits=pi_prime.on_logits,
#                 entropy_beta=self.model_beta,
#                 epsilon=clip_epsilon,
#                 name='on_policy',
#                 verbose=verbose
#             )
#
#             # Classification loss for price movements prediction:
#
#             # oracle_labels = tf.one_hot(tf.argmax(pi.expert_actions, axis=-1), depth=4)
#
#             if self.class_use_rnn:
#                 class_logits = pi.on_logits
#
#             else:
#                 class_logits = pi.on_simple_logits
#
#
#             # class_loss = tf.reduce_mean(
#             #     tf.nn.softmax_cross_entropy_with_logits_v2(
#             #         labels=pi.expert_actions,#oracle_labels,
#             #         logits=class_logits,
#             #     )
#             # )
#
#             class_loss = tf.losses.mean_squared_error(
#                 labels=pi.expert_actions[..., 1:3],
#                 predictions=tf.nn.softmax(class_logits)[..., 1:3],
#             )
#             entropy = tf.reduce_mean(cat_entropy(class_logits))
#
#             # self.accuracy = tf.metrics.accuracy(
#             #     labels=tf.argmax(pi.expert_actions, axis=-1),
#             #     predictions=tf.argmax(class_logits, axis=-1)
#             # )
#
#             self.accuracy = tf.metrics.accuracy(
#                 labels=tf.argmax(pi.expert_actions[..., 1:3], axis=-1),
#                 predictions=tf.argmax(class_logits[..., 1:3], axis=-1)
#             )
#
#             model_summaries = [
#                 tf.summary.scalar('class_loss', class_loss),
#                 tf.summary.scalar('class_accuracy', self.accuracy[0])
#             ]
#             # Accumulate total loss:
#             loss = float(self.class_lambda) * class_loss + float(self.aac_lambda) * on_pi_loss\
#                 - float(self.model_beta) * entropy
#
#             model_summaries += on_pi_summaries
#
#         return loss, model_summaries
#
#     def _make_train_op(self, pi, pi_prime, pi_global):
#         """
#         Defines training op graph and supplementary sync operations.
#
#         Args:
#             pi:                 policy network obj.
#             pi_prime:           optional policy network obj.
#             pi_global:          shared policy network obj. hosted by parameter server
#
#         Returns:
#             tensor holding training op graph;
#         """
#
#         # Each worker gets a different set of adam optimizer parameters:
#         self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)
#
#         # Clipped gradients:
#         self.grads, _ = tf.clip_by_global_norm(
#             tf.gradients(self.loss, pi.var_list),
#             40.0
#         )
#         self.grads_global_norm = tf.global_norm(self.grads)
#         # Copy weights from the parameter server to the local model
#         self.sync = self.sync_pi = tf.group(
#             *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
#         )
#         if self.use_target_policy:
#             # Copy weights from new policy model to target one:
#             self.sync_pi_prime = tf.group(
#                 *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
#             )
#         grads_and_vars = list(zip(self.grads, pi_global.var_list))
#
#         # Set global_step increment equal to observation space batch size:
#         obs_space_keys = list(pi.on_state_in.keys())
#
#         assert 'external' in obs_space_keys, \
#             'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
#         if isinstance(pi.on_state_in['external'], dict):
#             stream = pi.on_state_in['external'][list(pi.on_state_in['external'].keys())[0]]
#         else:
#             stream = pi.on_state_in['external']
#         self.inc_step = self.global_step.assign_add(tf.shape(stream)[0])
#
#         train_op = [self.optimizer.apply_gradients(grads_and_vars),  self.accuracy]
#
#         self.evaluate_op = [self.loss, self.accuracy]
#
#         self.log.debug('train_op defined')
#         return train_op
#
#     def _combine_summaries(self, policy=None, model_summaries=None):
#         """
#         Defines model-wide and episode-related summaries
#
#         Returns:
#             model_summary op
#             episode_summary op
#         """
#         if model_summaries is not None:
#             if self.use_global_network:
#                 # Model-wide statistics:
#                 with tf.name_scope('model'):
#                     model_summaries += [
#                         tf.summary.scalar("grad_global_norm", self.grads_global_norm),
#                         tf.summary.scalar("learn_rate", self.learn_rate_decayed),  # cause actual rate is a jaggy due to test freezes
#                         tf.summary.scalar("total_loss", self.loss),
#                     ]
#                     if policy is not None:
#                         model_summaries += [ tf.summary.scalar("var_global_norm", tf.global_norm(policy.var_list))]
#         else:
#             model_summaries = []
#         # Model stat. summary:
#         model_summary = tf.summary.merge(model_summaries, name='model_summary')
#
#         # Episode-related summaries:
#         ep_summary = dict(
#             # Summary placeholders
#             render_atari=tf.placeholder(tf.uint8, [None, None, None, 1]),
#             total_r=tf.placeholder(tf.float32, ),
#             cpu_time=tf.placeholder(tf.float32, ),
#             final_value=tf.placeholder(tf.float32, ),
#             steps=tf.placeholder(tf.int32, ),
#         )
#         if self.test_mode:
#             # For Atari:
#             ep_summary['render_op'] = tf.summary.image("model/state", ep_summary['render_atari'])
#
#         else:
#             # BTGym rendering:
#             ep_summary.update(
#                 {
#                     mode: tf.placeholder(tf.uint8, [None, None, None, None], name=mode + '_pl')
#                     for mode in self.env_list[0].render_modes + self.aux_render_modes
#                 }
#             )
#             ep_summary['render_op'] = tf.summary.merge(
#                 [tf.summary.image(mode, ep_summary[mode])
#                  for mode in self.env_list[0].render_modes + self.aux_render_modes]
#             )
#         # Episode stat. summary:
#         ep_summary['btgym_stat_op'] = tf.summary.merge(
#             [
#                 tf.summary.scalar('episode_train/cpu_time_sec', ep_summary['cpu_time']),
#                 tf.summary.scalar('episode_train/total_reward', ep_summary['total_r']),
#             ],
#             name='episode_train_btgym'
#         )
#         # Test episode stat. summary:
#         ep_summary['test_btgym_stat_op'] = tf.summary.merge(
#             [
#                 tf.summary.scalar('episode_test/total_reward', ep_summary['total_r']),
#             ],
#             name='episode_test_btgym'
#         )
#         ep_summary['atari_stat_op'] = tf.summary.merge(
#             [
#                 tf.summary.scalar('episode/total_reward', ep_summary['total_r']),
#                 tf.summary.scalar('episode/steps', ep_summary['steps'])
#             ],
#             name='episode_atari'
#         )
#         self.log.debug('model-wide and episode summaries ok.')
#         return model_summary, ep_summary
#
#     def get_sample_config(self, **kwargs):
#         """
#         Returns environment configuration parameters for next episode to sample.
#
#         Here we always prescribe to sample test episode from source or target domain.
#
#         Args:
#               kwargs:     not used
#
#         Returns:
#             configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
#         """
#
#         new_trial = 0
#         if self.current_episode >= self.num_episodes_per_trial:
#             # Reset episode counter:
#             self.current_episode = 0
#
#             # Request new trial:
#             new_trial = 1
#             # Decide on trial type (source/target):
#             if self.current_source_trial >= self.num_source_trials:
#                 # Time to switch to target mode:
#                 self.current_trial_mode = 1
#                 # Reset counters:
#                 self.current_source_trial = 0
#                 self.current_target_trial = 0
#
#             if self.current_target_trial >= self.num_target_trials:
#                 # Vise versa:
#                 self.current_trial_mode = 0
#                 self.current_source_trial = 0
#                 self.current_target_trial = 0
#
#             # Update counter:
#             if self.current_trial_mode:
#                 self.current_target_trial += 1
#             else:
#                 self.current_source_trial += 1
#
#         self.current_episode += 1
#
#         if self.task == 0:
#             trial_sample_type = 1
#
#         else:
#             trial_sample_type = self.current_trial_mode
#
#         # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
#         sample_config = dict(
#             episode_config=dict(
#                 get_new=True,
#                 sample_type=1,
#                 timestamp= self.global_timestamp,
#                 b_alpha=self.episode_sample_params[0],
#                 b_beta=self.episode_sample_params[-1]
#             ),
#             trial_config=dict(
#                 get_new=new_trial,
#                 sample_type=trial_sample_type,
#                 timestamp=self.global_timestamp,
#                 b_alpha=self.trial_sample_params[0],
#                 b_beta=self.trial_sample_params[-1]
#             )
#         )
#         return sample_config
#
#     def process(self, sess, **kwargs):
#         try:
#             sess.run(self.sync_pi)
#             # Get data configuration:
#             data_config = self.get_sample_config()
#
#             # self.log.warning('data_config: {}'.format(data_config))
#
#             # If this step data comes from source or target domain
#             is_test = data_config['trial_config']['sample_type'] and data_config['episode_config']['sample_type']
#
#             # self.log.warning('is_test: {}'.format(is_test))
#
#             if is_test:
#                 if self.task == 0:
#                     self.process_eval(sess, data_config)
#
#                 else:
#                     pass
#
#             else:
#                 self.process_train(sess, data_config)
#
#         except:
#             msg = 'process() exception occurred' + \
#                 '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
#             self.log.exception(msg)
#             raise RuntimeError(msg)
#
#     def process_eval(self, sess, data_config):
#         data = {}
#         done = False
#         # # Set target episode beginning to be at current timepoint:
#         data_config['trial_config']['align_left'] = 0
#         self.log.info('test episode started...')
#
#         while not done:
#             sess.run(self.sync_pi)
#
#             data = self.get_data(
#                 policy=self.local_network,
#                 data_sample_config=data_config,
#             )
#             done = np.asarray(data['terminal']).any()
#             feed_dict = self.process_data(sess, data, is_train=False, pi=self.local_network)
#
#             fetches = [self.evaluate_op, self.model_summary_op, self.inc_step]
#             fetched = sess.run(fetches, feed_dict=feed_dict)
#
#             model_summary = fetched[-2]
#
#             self.process_summary(sess, data, model_summary)
#
#             # self.global_timestamp = data['on_policy'][0]['state']['metadata']['timestamp'][-1]
#
#         self.log.info(
#             'test episode finished, global_time set: {}'.format(
#                 datetime.datetime.fromtimestamp(self.global_timestamp)
#             )
#         )
#
#     def process_train(self, sess, data_config):
#         data = {}
#         done = False
#         # Set source episode to be sampled uniformly from test interval:
#         data_config['trial_config']['align_left'] = 0
#         # self.log.warning('train episode started...')
#
#         while not done:
#             sess.run(self.sync_pi)
#
#             wirte_model_summary = \
#                 self.local_steps % self.model_summary_freq == 0
#
#             data = self.get_data(
#                 policy=self.local_network,
#                 data_sample_config=data_config
#             )
#             done = np.asarray(data['terminal']).any()
#             feed_dict = self.process_data(sess, data, is_train=True, pi=self.local_network)
#
#             if wirte_model_summary:
#                 fetches = [self.train_op, self.model_summary_op, self.inc_step]
#             else:
#                 fetches = [self.train_op, self.inc_step]
#
#             fetched = sess.run(fetches, feed_dict=feed_dict)
#
#             if wirte_model_summary:
#                 model_summary = fetched[-2]
#
#             else:
#                 model_summary = None
#
#             self.process_summary(sess, data, model_summary)
#
#             self.local_steps += 1
#
#         self.log.info(
#             'train episode finished at {} vs was_global_time: {}'.format(
#                 data['on_policy'][0]['info']['time'][-1],
#                 datetime.datetime.fromtimestamp(data['on_policy'][0]['state']['metadata']['timestamp'][-1])
#
#             )
#         )


class RegressionTestAAC(BaseAAC):
    """
    Simplified AAC class meant to test state encoder ability to solve an isolated classification/regression problem.
    """
    def __init__(
            self,
            runner_config=None,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,
            test_slowdown_steps=0,
            episode_sample_params=(1.0, 1.0),
            trial_sample_params=(1.0, 1.0),
            aac_lambda=0,
            regress_lambda=1.0,
            aux_render_modes=(),
            _use_target_policy=False,
            name='TestAAC',
            **kwargs
    ):
        try:
            if runner_config is None:
                self.runner_config = {
                    'class_ref': RegressionRunner,
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

            self.aac_lambda = aac_lambda
            self.regress_lambda = regress_lambda

            self.test_slowdown_steps = test_slowdown_steps

            self.episode_sample_params = episode_sample_params
            self.trial_sample_params = trial_sample_params

            self.global_timestamp = 0

            self.current_source_trial = 0
            self.current_target_trial = 0
            self.current_trial_mode = 0  # source
            self.current_episode = 1

            super(RegressionTestAAC, self).__init__(
                runner_config=self.runner_config,
                aux_render_modes=aux_render_modes,
                name=name,
                **kwargs
            )
        except:
            msg = '{}.__init()__ exception occurred'.format(name) + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_loss(self, pi, pi_prime, name='base', verbose=True, **kwargs):
        """
        Defines policy state encoder regression loss, placeholders and summaries.

        Args:
            pi:                 policy network obj.
            pi_prime:           optional policy network obj.
            name:               str, name scope
            verbose:            summary level

        Returns:
            tensor holding estimated loss graph
            list of related summaries
        """
        with tf.name_scope(name):
            # On-policy AAC loss definition:
            pi.on_pi_act_target = tf.placeholder(
                tf.float32, [None, self.ref_env.action_space.one_hot_depth], name="on_policy_action_pl"
            )
            pi.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            pi.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            # clip_epsilon = tf.cast(self.clip_epsilon * self.learn_rate_decayed / self.opt_learn_rate, tf.float32)
            #
            # on_pi_loss, on_pi_summaries = self.on_policy_loss(
            #     act_target=pi.on_pi_act_target,
            #     adv_target=pi.on_pi_adv_target,
            #     r_target=pi.on_pi_r_target,
            #     pi_logits=pi.on_logits,
            #     pi_vf=pi.on_vf,
            #     pi_prime_logits=pi_prime.on_logits,
            #     entropy_beta=self.model_beta,
            #     epsilon=clip_epsilon,
            #     name='on_policy',
            #     verbose=verbose
            # )
            pi_regression = tf.exp(pi.regression)
            regress_loss = tf.losses.mean_squared_error(
                labels=pi.regression_targets,
                predictions=pi_regression,
                weights=self.regress_lambda,
            )

            self.mse = tf.metrics.mean_squared_error(
                labels=pi.regression_targets,
                predictions=pi_regression
            )

            model_summaries = [
                tf.summary.scalar('regress_loss', regress_loss),
                tf.summary.scalar('mse_metric', self.mse[0])
            ]
            # Accumulate total loss:
            # loss = float(self.class_lambda) * regress_loss + float(self.aac_lambda) * on_pi_loss\
            #     - float(self.model_beta) * entropy

            #model_summaries += on_pi_summaries

            loss = regress_loss

            return loss, model_summaries

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

        # Copy weights from the parameter server to the local model:
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

        required_modes = ['external', 'regression_targets', 'metadata']
        for mode in required_modes:
            assert mode in obs_space_keys, \
                'Expected observation space to contain `{}` mode, got: {}'.format(mode, obs_space_keys)

        if isinstance(pi.on_state_in['external'], dict):
            stream = pi.on_state_in['external'][list(pi.on_state_in['external'].keys())[0]]
        else:
            stream = pi.on_state_in['external']
        self.inc_step = self.global_step.assign_add(tf.shape(stream)[0])

        train_op = [self.optimizer.apply_gradients(grads_and_vars), self.mse]

        self.log.debug('train_op defined')
        return train_op

    def _combine_summaries(self, policy=None, model_summaries=None):
        """
        Defines model-wide and episode-related summaries

        Returns:
            model_summary op
            episode_summary op
        """
        if model_summaries is not None:
            if self.use_global_network:
                # Model-wide statistics:
                with tf.name_scope('model'):
                    model_summaries += [
                        tf.summary.scalar("grad_global_norm", self.grads_global_norm),
                        tf.summary.scalar("learn_rate", self.learn_rate_decayed),
                        # cause actual rate is a jaggy due to test freezes
                        tf.summary.scalar("total_loss", self.loss),
                    ]
                    if policy is not None:
                        model_summaries += [tf.summary.scalar("var_global_norm", tf.global_norm(policy.var_list))]
        else:
            model_summaries = []
        # Model stat. summary:
        model_summary = tf.summary.merge(model_summaries, name='model_summary')

        # Episode-related summaries:
        ep_summary = dict(
            # Summary placeholders
            render_atari=tf.placeholder(tf.uint8, [None, None, None, 1]),
            total_r=tf.placeholder(tf.float32, ),
            cpu_time=tf.placeholder(tf.float32, ),
            final_value=tf.placeholder(tf.float32, ),
            steps=tf.placeholder(tf.int32, ),
        )
        if self.test_mode:
            # For Atari:
            ep_summary['render_op'] = tf.summary.image("model/state", ep_summary['render_atari'])

        else:
            # BTGym rendering:
            ep_summary.update(
                {
                    mode: tf.placeholder(tf.uint8, [None, None, None, None], name=mode + '_pl')
                    for mode in self.env_list[0].render_modes + self.aux_render_modes
                }
            )
            ep_summary['render_op'] = tf.summary.merge(
                [tf.summary.image(mode, ep_summary[mode])
                 for mode in self.env_list[0].render_modes + self.aux_render_modes]
            )
        # Episode stat. summary:
        ep_summary['btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_train/cpu_time_sec', ep_summary['cpu_time']),
                tf.summary.scalar('episode_train/total_reward', ep_summary['total_r']),
            ],
            name='episode_train_btgym'
        )
        # Test episode stat. summary:
        ep_summary['test_btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_test/total_reward', ep_summary['total_r']),
            ],
            name='episode_test_btgym'
        )
        ep_summary['atari_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode/steps', ep_summary['steps'])
            ],
            name='episode_atari'
        )
        self.log.debug('model-wide and episode summaries ok.')
        return model_summary, ep_summary

    def process(self, sess, **kwargs):
        try:
            sess.run(self.sync_pi)
            # Get data configuration:
            data_config = self.get_sample_config()

            self.process_train(sess, data_config)

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def process_train(self, sess, data_config):
        data = {}
        done = False
        # Set source episode to be sampled uniformly from test interval:
        data_config['trial_config']['align_left'] = 0
        # self.log.warning('train episode started...')

        while not done:
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

        self.log.info(
            'train episode finished at {} vs was_global_time: {}'.format(
                data['on_policy'][0]['info']['time'][-1],
                datetime.datetime.fromtimestamp(data['on_policy'][0]['state']['metadata']['timestamp'][-1])

            )
        )
