import tensorflow as tf
import numpy as np
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class AMLDG_1(GuidedAAC):
    """
    Asynchronous implementation of MLDG algorithm (by Da Li et al.)
    for one-shot adaptation in dynamically changing environments.

    MOD: Defines meta-test task as one-roll-ahead of train one.

    Does not relies on sub-AAC classes.

    TODO: Use per-episode replay buffer distribution support as in AMLDG_d instead of single previous rollout

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

            super(AMLDG_1, self).__init__(
                runner_config=self.runner_config,
                _use_target_policy=True,
                _aux_render_modes=_aux_render_modes,
                name=name,
                **kwargs
            )
            self.model_summary_op = tf.summary.merge(
                [self.model_summary_op, self._combine_meta_summaries()],
                name='meta_model_summary'
            )
        except:
            msg = '{}.__init()__ exception occurred'.format(name) + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_loss(self, pi, pi_prime):
        self.meta_train_loss, meta_train_summaries = self._make_base_loss(
            pi=pi,
            pi_prime=self.dummy_pi,
            name=self.name + '/meta_train',
            verbose=True
        )
        self.meta_test_loss, meta_test_summaries = self._make_base_loss(
            pi=pi_prime,
            pi_prime=self.dummy_pi,
            name=self.name + '/meta_test',
            verbose=True
        )
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

        # Guided losses, need two of them:
        # guided_train_loss, _ = self.expert_loss(
        #     pi_actions=pi.on_logits,
        #     expert_actions=pi.expert_actions,
        #     name='on_policy',
        #     verbose=False,
        #     guided_lambda=self.train_guided_lambda
        # )
        guided_test_loss, g_summary = self.expert_loss(
            pi_actions=pi_prime.on_logits,
            expert_actions=pi_prime.expert_actions,
            name='on_policy',
            verbose=True,
            guided_lambda=self.train_guided_lambda
        )

        # self.meta_train_loss += guided_train_loss
        self.meta_test_loss += guided_test_loss

        return self.meta_train_loss + self.meta_test_loss, meta_train_summaries + meta_test_summaries + g_summary

    def _make_train_op(self, pi, pi_prime, pi_global):
        """
        Defines training op graph and supplementary sync operations.

        Returns:
            tensor holding training op graph;
        """
        # Copy weights from the parameter server to the local pi:
        self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
        )
        # From ps to pi_prime:
        self.sync_pi_prime = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi_global.var_list)]
        )
        # From pi_prime to pi:
        self.sync_pi_from_prime = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_prime.var_list)]
        )
        self.sync = [self.sync_pi, self.sync_pi_prime]
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)
        self.fast_optimizer = tf.train.GradientDescentOptimizer(self.fast_opt_learn_rate)

        # Clipped gradients:
        pi.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.meta_train_loss, pi.var_list),
            40.0
        )
        pi_prime.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.meta_test_loss, pi_prime.var_list),
            40.0
        )
        # Meta_optimisation gradients as sum of meta-train and meta-test gradients:
        self.grads = []
        for g1, g2 in zip(pi.grads, pi_prime.grads):
            if g1 is not None and g2 is not None:
                meta_g = g1 + g2  # if g1 is excluded - we got MAML

            else:
                meta_g = None  # need this to map correctly to vars

            self.grads.append(meta_g)

        # Gradients to update local meta-test policy (conditioned on train data):
        train_grads_and_vars = list(zip(pi.grads, pi_prime.var_list))

        # Meta-gradients to be sent to parameter server:
        meta_grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # Remove empty entries:
        meta_grads_and_vars = [(g, v) for (g, v) in meta_grads_and_vars if g is not None]

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(pi.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(self.local_network.on_state_in['external'])[0])

        # Local fast optimisation op:
        self.fast_train_op = self.fast_optimizer.apply_gradients(train_grads_and_vars)

        # Global meta-optimisation op:
        self.meta_train_op = self.optimizer.apply_gradients(meta_grads_and_vars)

        self.log.debug('train_op defined')
        return self.fast_train_op, self.meta_train_op

    def _combine_meta_summaries(self):
        """
        Additional summaries here.
        """
        with tf.name_scope(self.name):
            meta_model_summaries = [
                tf.summary.scalar('meta_grad_global_norm', tf.global_norm(self.grads)),
                # tf.summary.scalar('total_meta_loss', self.loss),
                # tf.summary.scalar('alpha_learn_rate', self.alpha_rate),
                # tf.summary.scalar('alpha_learn_rate_loss', self.alpha_rate_loss)
            ]
        return meta_model_summaries

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

    def _process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            sess.run(self.sync_pi)
            sess.run(self.sync_pi_prime)

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

                feed_dict = self.process_data(sess, train_data, is_train=is_train, pi=self.local_network)

                fetches = [self.fast_train_op]

                fetched = sess.run(fetches, feed_dict=feed_dict)

                # self.log.warning('Train gradients ok.')

                # Collect test rollout using [updated] pi_prime policy:
                test_data = self.get_data(
                    policy=self.local_network_prime,
                    data_sample_config=data_config
                )

                # self.log.debug('test_rollout_ok')

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                # TODO: paranoid check is_train ~ actual_data_trial_type

                if is_train:
                    # Process test data and perform meta-optimisation step:
                    feed_dict.update(
                        self.process_data(sess, test_data, is_train=True, pi=self.local_network_prime)
                    )

                    if wirte_model_summary:
                        meta_fetches = [self.meta_train_op, self.model_summary_op, self.inc_step]
                    else:
                        meta_fetches = [self.meta_train_op, self.inc_step]

                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                    # self.log.warning('Meta-gradients ok.')
                else:
                    # True test, no updates sent to parameter server:
                    meta_fetched = [None, None]

                    # self.log.warning('Meta-opt. rollout ok.')

                if wirte_model_summary:
                    meta_model_summary = meta_fetched[-2]
                    model_summary = fetched[-1]

                else:
                    meta_model_summary = None
                    model_summary = None

                # Next step housekeeping:
                sess.run(self.sync_pi_from_prime)

                # TODO: ????
                # sess.run(self.sync_pi_prime)

                # Make this test trajectory next train:
                train_data = test_data
                # self.log.warning('Trajectories swapped.')

                # Write down summaries:
                self.process_summary(sess, test_data, meta_model_summary)

                self.local_steps += 1
                roll_num += 1

        except:
            msg = '.process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

