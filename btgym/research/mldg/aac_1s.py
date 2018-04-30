import tensorflow as tf
import numpy as np

from btgym.algorithms.utils import batch_stack, batch_gather
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner
from btgym.research.mldg.aac_1 import AMLDG_1
from btgym.research.mldg.memory import LocalMemory2


class AMLDG_1s(GuidedAAC):
    """
    AMLDG_1 + t2d methods + another style of update
    """

    def __init__(
            self,
            num_train_updates=1,
            train_batch_size=64,
            runner_config=None,
            fast_opt_learn_rate=1e-3,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='AMLDG1s',
            **kwargs
         ):
        if runner_config is None:
            self.runner_config = {
                'class_ref': BaseSynchroRunner,
                'kwargs': {
                    'data_sample_config': {'mode': 0},
                    'test_conditions': {
                        'state': {
                            'metadata': {
                                'trial_type': 1  # any type of episode from target dom. considered test one
                            }
                        }
                    },
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
        self.train_batch_size = train_batch_size
        self.num_train_updates = num_train_updates
        self.episode_memory = LocalMemory2()

        super().__init__(
            runner_config=self.runner_config,
            use_off_policy_aac=True,
            _use_target_policy=False,
            _use_local_memory=True,
            _aux_render_modes=_aux_render_modes,
            name=name,
            **kwargs
        )

    def half_process_data(self, sess, data, is_train, pi, pi_prime=None):
        """
        Processes data but returns batched data instead of train step feed dictionary.
        Args:
            sess:               tf session obj.
            pi:                 policy to feed
            pi_prime:           optional policy to feed
            data (dict):        data dictionary
            is_train (bool):    is data provided are train or test

        Returns:
            feed_dict (dict):   train step feed dictionary
        """
        # Process minibatch for on-policy train step:
        on_policy_batch = self._process_rollouts(data['on_policy'])

        if self.use_memory:
            # Process rollouts from replay memory:
            off_policy_batch = self._process_rollouts(data['off_policy'])

            if self.use_reward_prediction:
                # Rebalanced 50/50 sample for RP:
                rp_rollouts = data['off_policy_rp']
                rp_batch = batch_stack([rp.process_rp(self.rp_reward_threshold) for rp in rp_rollouts])

            else:
                rp_batch = None

        else:
            off_policy_batch = None
            rp_batch = None

        return {
            'on_policy_batch': on_policy_batch,
            'off_policy_batch': off_policy_batch,
            'rp_batch': rp_batch
        }

    @staticmethod
    def _check(batch):
        """
        Debug. utility.
        """
        print('Got data_dict:')
        for key in batch.keys():
            try:
                shape = np.asarray(batch[key]).shape
            except:
                shape = '???'
            print('key: {}, shape: {}'.format(key, shape))

    def get_sample_config(self, **kwargs):
        """
        Returns environment configuration parameters for next episode to sample.

        Always prescribes to sample train episode from source or target domain.

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

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=0,
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

    def start(self, sess, summary_writer, **kwargs):
        """
        Executes all initializing operations,
        starts environment runner[s].
        Supposed to be called by parent worker just before training loop starts.

        Args:
            sess:           tf session object.
            kwargs:         not used by default.
        """
        try:
            # Copy weights from global to local:
            sess.run(self.sync)

            # Start thread_runners:
            self._start_runners(
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.get_sample_config(mode=1)
            )

            self.summary_writer = summary_writer
            self.log.notice('runner started.')

        except:
            msg = '.start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_loss(self, pi, pi_prime, name='base', verbose=True):
        """
        Defines base AAC on- and off-policy loss, optional VR loss[not used yet], placeholders and summaries.

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

            guided_test_loss, g_summary = self.expert_loss(
                pi_actions=pi.on_logits,
                expert_actions=pi.expert_actions,
                name='on_policy',
                verbose=True,
                guided_lambda=self.train_guided_lambda
            )

            # On-policy AAC loss definition:
            pi.on_pi_act_target = tf.placeholder(
                tf.float32, [None, self.ref_env.action_space.n], name="on_policy_action_pl"
            )
            pi.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            pi.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            clip_epsilon = tf.cast(self.clip_epsilon * self.learn_rate_decayed / self.opt_learn_rate, tf.float32)

            self.on_pi_loss, on_pi_summaries = self.on_policy_loss(
                act_target=pi.on_pi_act_target,
                adv_target=pi.on_pi_adv_target,
                r_target=pi.on_pi_r_target,
                pi_logits=pi.on_logits,
                pi_vf=pi.on_vf,
                pi_prime_logits=pi_prime.on_logits,
                entropy_beta=self.model_beta,
                epsilon=clip_epsilon,
                name='on_policy',
                verbose=verbose
            )
            self.on_pi_loss += guided_test_loss
            model_summaries = on_pi_summaries + g_summary

            # Off-policy losses:
            pi.off_pi_act_target = tf.placeholder(
                tf.float32, [None, self.ref_env.action_space.n], name="off_policy_action_pl")
            pi.off_pi_adv_target = tf.placeholder(tf.float32, [None], name="off_policy_advantage_pl")
            pi.off_pi_r_target = tf.placeholder(tf.float32, [None], name="off_policy_return_pl")

            if self.use_off_policy_aac:
                # Off-policy AAC loss graph mirrors on-policy:
                self.off_pi_loss, self.off_pi_summaries = self.off_policy_loss(
                    act_target=pi.off_pi_act_target,
                    adv_target=pi.off_pi_adv_target,
                    r_target=pi.off_pi_r_target,
                    pi_logits=pi.off_logits,
                    pi_vf=pi.off_vf,
                    pi_prime_logits=pi_prime.off_logits,
                    entropy_beta=self.model_beta,
                    epsilon=clip_epsilon,
                    name='off_policy',
                    verbose=False
                )
                self.off_pi_loss *= self.off_aac_lambda

            if self.use_value_replay:
                # Value function replay loss:
                pi.vr_target = tf.placeholder(tf.float32, [None], name="vr_target")
                self.vr_loss, self.vr_summaries = self.vr_loss(
                    r_target=pi.vr_target,
                    pi_vf=pi.vr_value,
                    name='off_policy',
                    verbose=verbose
                )

        return self.on_pi_loss, model_summaries

    def _make_train_op(self, pi, pi_prime, pi_global):
        """
        Defines training op graph and supplementary sync operations.
            TODO:
                - joint on-policy(~meta-test) and sampled off-policy (~meta-train) optimization:
                single optimizer, same learn rate;

                - separate optimization with different rates;

        Returns:
            tensor holding training op graph;
        """
        # Copy weights from the parameter server to the local pi:
        self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
        )
        self.sync = self.sync_pi
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)
        self.fast_optimizer = tf.train.GradientDescentOptimizer(self.fast_opt_learn_rate)

        # Clipped gradients:
        pi.on_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.on_pi_loss, pi.var_list),
            40.0
        )
        pi.off_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.off_pi_loss, pi.var_list),
            40.0
        )

        pi.on_grads = [
            g1 + g2 if g1 is not None and g2 is not None else g1 if g2 is None else g2
            for g1, g2 in zip(pi.on_grads, pi.off_grads)
        ]

        self.grads = pi.on_grads

        # Gradients to update local policy (conditioned on train, off-policy data):
        local_grads_and_vars = list(zip(pi.off_grads, pi.var_list))

        # Meta-gradients to be sent to parameter server:
        global_grads_and_vars = list(zip(pi.on_grads, pi_global.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(pi.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in['external'])[0])

        # Local fast optimisation op:
        self.local_train_op = self.fast_optimizer.apply_gradients(local_grads_and_vars)

        # Global meta-optimisation op:
        self.global_train_op = self.optimizer.apply_gradients(global_grads_and_vars)

        self.log.debug('train_op defined')
        return [self.local_train_op, self.global_train_op]

    def _process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            sess.run(self.sync_pi)

            self.episode_memory.reset()

            # Get data configuration,
            data_config = self.get_sample_config(mode=1)

            # self.log.warning('data_config: {}'.format(data_config))

            # If this step data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_train = not data_config['trial_config']['sample_type']
            done = False
            roll_num = 0

            #  ** Data leakage checks removed.

            # Collect initial trajectory rollout:
            train_data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config,
                force_new_episode=True
            )

            # self.log.warning('initial_rollout_ok')

            while not done:
                # self.log.warning('Roll #{}'.format(roll_num))

                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                self.episode_memory.add_batch(
                    **self.half_process_data(sess, train_data, is_train=is_train, pi=self.local_network)
                )
                # self.log.warning('train rollout added to memory.')

                # Collect test rollout using [updated by prev. step] policy:
                test_data = self.get_data(
                    policy=self.local_network,
                    data_sample_config=data_config
                )

                # self.log.warning('test rollout collected.')

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                # TODO: paranoid check is_train ~ actual_data_trial_type

                feed_dict = self.process_data(sess, test_data, is_train=is_train, pi=self.local_network)

                # self._check(feed_dict)

                feed_dict.update(
                    self._get_main_feeder(
                        sess,
                        **self.episode_memory.sample(self.train_batch_size),
                        is_train=is_train,
                        pi=self.local_network,
                    )
                )

                # self._check(feed_dict)

                if is_train:
                    train_op = [self.local_train_op, self.train_op]

                else:
                    train_op = [self.local_train_op]

                if wirte_model_summary:
                    meta_fetches = train_op + [self.model_summary_op, self.inc_step]
                else:
                    meta_fetches = train_op + [self.inc_step]

                meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                if wirte_model_summary and is_train:
                    meta_model_summary = meta_fetched[-2]

                else:
                    meta_model_summary = None

                # Make this test trajectory next train:
                train_data = test_data
                # self.log.warning('Trajectories swapped.')

                # Write down summaries:
                self.process_summary(sess, test_data, model_data=meta_model_summary)

                self.local_steps += 1
                roll_num += 1

        except:
            msg = '.process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


class AMLDG_1s_a(AMLDG_1s):
    """
    FAILED
    """

    def __init__(self, num_train_updates = 1, name='AMLDG1sa', **kwargs):
        super(AMLDG_1s_a, self).__init__(name=name, **kwargs)

        self.num_train_updates = num_train_updates
        # self.meta_summaries = self.combine_aux_summaries()

    def _make_train_op(self, pi, pi_prime, pi_global):
        """
        Defines training op graph and supplementary sync operations.

                - separate optimization with different rates;

        Returns:
            tensor holding training op graph;
        """
        # Copy weights from the parameter server to the local pi:
        self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
        )
        self.sync = self.sync_pi
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)

        # Clipped gradients:
        pi.on_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.on_pi_loss, pi.var_list),
            40.0
        )
        pi.off_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.off_pi_loss, pi.var_list),
            40.0
        )
        self.grads = pi.on_grads

        # Learnable fast rate:
        #self.fast_learn_rate = tf.reduce_mean(pi.off_learn_alpha, name='mean_alpha_rate') / 10
        self.fast_optimizer = tf.train.GradientDescentOptimizer(self.fast_opt_learn_rate)
        # self.alpha_rate_loss = tf.global_norm(pi.off_grads)
        # self.alpha_grads, _ = tf.clip_by_global_norm(
        #     tf.gradients(self.alpha_rate_loss, pi.var_list),
        #     40.0
        # )
        # # Sum on_ and  second order alpha_ gradients:
        # pi.off_grads = [
        #     g1 + g2 if g1 is not None and g2 is not None else g1 if g2 is None else g2
        #     for g1, g2 in zip(pi.off_grads, self.alpha_grads)
        # ]

        # Gradients to update local policy (conditioned on train, off-policy data):
        local_grads_and_vars = list(zip(pi.off_grads, pi.var_list))

        # Meta-gradients to be sent to parameter server:
        global_grads_and_vars = list(zip(pi.on_grads, pi_global.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(pi.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in['external'])[0])

        # Local fast optimisation op:
        self.local_train_op = self.fast_optimizer.apply_gradients(local_grads_and_vars)

        # Global meta-optimisation op:
        self.global_train_op = self.optimizer.apply_gradients(global_grads_and_vars)

        self.log.debug('train_op defined')
        return [self.local_train_op, self.global_train_op]

    # def combine_aux_summaries(self):
    #     """
    #     Additional summaries here.
    #     """
    #     off_model_summaries = tf.summary.merge(
    #         [
    #             tf.summary.scalar('alpha_rate', self.fast_learn_rate),
    #         ]
    #     )
    #     return off_model_summaries

    def _process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            sess.run(self.sync_pi)

            self.episode_memory.reset()

            # Get data configuration,
            data_config = self.get_sample_config(mode=1)

            # self.log.warning('data_config: {}'.format(data_config))

            # If this step data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_train = not data_config['trial_config']['sample_type']
            done = False
            roll_num = 0

            #  ** Data leakage checks removed.

            # Collect initial trajectory rollout:
            train_data = self.get_data(
                policy=self.local_network,
                data_sample_config=data_config,
                force_new_episode=True
            )

            # self.log.warning('initial_rollout_ok')

            while not done:
                # self.log.warning('Roll #{}'.format(roll_num))

                write_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                # Add to replay buffer:
                self.episode_memory.add_batch(
                    **self.half_process_data(sess, train_data, is_train=is_train, pi=self.local_network)
                )
                # self.log.warning('train rollout added to memory.')
                feed_dict = {}
                for i in range(self.num_train_updates):
                    # Sample off policy data and make feeder:
                    feed_dict = (
                        self._get_main_feeder(
                            sess,
                            **self.episode_memory.sample(self.train_batch_size),
                            is_train=is_train,
                            pi=self.local_network,
                        )
                    )
                    # self._check(feed_dict)

                    fetches = [self.local_train_op]

                    fetched = sess.run(fetches, feed_dict=feed_dict)

                    # # Write down particular model summary:
                    # if write_model_summary:
                    #     self.summary_writer.add_summary(tf.Summary.FromString(fetched[-1]), sess.run(self.global_step))
                    #     self.summary_writer.flush()

                # Collect test on_policy rollout using [updated] policy:
                test_data = self.get_data(
                    policy=self.local_network,
                    data_sample_config=data_config
                )

                # self.log.warning('test rollout collected.')

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                # TODO: paranoid check is_train ~ actual_data_trial_type

                feed_dict.update(self.process_data(sess, test_data, is_train=is_train, pi=self.local_network))

                # self._check(feed_dict)

                if is_train:
                    train_op = self.train_op

                    if write_model_summary:
                        meta_fetches = [train_op, self.model_summary_op, self.inc_step]

                    else:
                        meta_fetches = [train_op, self.inc_step]

                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                else:
                    meta_fetched = [None, None]

                if write_model_summary:

                    meta_model_summary = meta_fetched[-2]

                else:
                    meta_model_summary = None

                # Make this test trajectory next train:
                train_data = test_data
                # self.log.warning('Trajectories swapped.')

                # Write down summaries:
                self.process_summary(sess, test_data, model_data=meta_model_summary)

                self.local_steps += 1
                roll_num += 1

        except:
            msg = '.process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


