#from btgym.algorithms.aac import BaseAAC
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner
import tensorflow as tf
import numpy as np


class FS_AAC_0(GuidedAAC):

    def __init__(
            self,
            runner_config=None,
            fast_learn_rate=0.1,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='SFAAC',
            **kwargs
    ):
        try:
            if runner_config is None:
                self.runner_config = {
                    'class_ref': BaseSynchroRunner,
                    'kwargs': {
                        'name': '',
                        'data_sample_config': {'mode': 0},
                    },
                }
            else:
                self.runner_config = runner_config

            self.fast_learn_rate = fast_learn_rate

            # Trials sampling control:
            self.num_source_trials = trial_source_target_cycle[0]
            self.num_target_trials = trial_source_target_cycle[-1]
            self.num_episodes_per_trial = num_episodes_per_trial

            self.current_source_trial = 0
            self.current_target_trial = 0
            self.current_trial_mode = 0  # source
            self.current_episode = 0

            super(FS_AAC_0, self).__init__(
                runner_config=self.runner_config,
                _aux_render_modes=_aux_render_modes,
                name=name,
                **kwargs
            )

        except:
            msg = 'SFAAC.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_sample_config(self, mode=1):
        """
        Returns environment configuration parameters for next episode to sample.

        Args:
              mode:     bool, False for slave, True for master

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """

        new_trial = 0
        mode = 1
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
                sample_type=self.current_trial_mode,
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
            sess.run(self.sync_pi)


            # Start thread_runners:
            self._start_runners(
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.get_sample_config(mode=1)
            )

            self.summary_writer = summary_writer
            self.log.notice('Runners started.')

        except:
            msg = 'start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_train_op(self):
        """
        Defines training op graph and supplementary sync operations.

        Returns:
            tensor holding training op graph;
        """

        # Each worker gets a different set of adam optimizer parameters:
        self.slow_optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)
        self.fast_optimizer = tf.train.GradientDescentOptimizer(self.fast_learn_rate)

        # Clipped gradients:
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
        grads_and_local_vars = list(zip(self.grads, self.local_network.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.local_network.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.inc_step = self.global_step.assign_add(tf.shape(self.local_network.on_state_in['external'])[0])

        train_op = self.slow_optimizer.apply_gradients(grads_and_vars)
        self.fast_train_op = self.fast_optimizer.apply_gradients(grads_and_local_vars)

        self.log.debug('train_op defined')
        return train_op

    def process(self, sess, **kwargs):
        """
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server:
            sess.run(self.sync_pi)

            # self.log.warning('Init Sync ok.')

            train_data_config = self.get_sample_config(mode=1)  # master env., samples trial

            # self.log.warning('train_data_config: {}'.format(train_data_config))

            # If this episode data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_train = not train_data_config['trial_config']['sample_type']
            # self.log.warning(
            #     'config: {}, is_train: {}'.format(train_data_config['trial_config']['sample_type'], is_train)
            # )
            done = False

            # Collect initial meta-train trajectory rollout:
            train_data = self.get_data(data_sample_config=train_data_config, force_new_episode=True)
            feed_dict = self.process_data(sess, train_data, is_train=is_train)

            # self.log.warning('Init Train data ok.')

            # Disable possibility of master data runner acquiring new trials,
            # in case meta-train episode terminates earlier than meta-test -
            # we than need to get additional meta-train trajectories from exactly same distribution (trial):
            train_data_config['trial_config']['get_new'] = 0

            roll_num = 0

            # Collect entire meta-test episode rollout by rollout:
            while not done:
                # self.log.warning('Roll #{}'.format(roll_num))

                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                # If episode has just ended?
                done = np.asarray(train_data['terminal']).any()

                if is_train:
                    # Update local pi(with fast_learn_rate) and global shared parameters (via slow_learn_rate):
                    if wirte_model_summary:
                        fetches = [
                            self.train_op,
                            #self.fast_train_op,
                            self.model_summary_op,
                            self.inc_step
                        ]
                    else:
                        fetches = [
                            self.train_op,
                            #self.fast_train_op,
                            self.inc_step
                        ]

                    fetched = sess.run(fetches, feed_dict=feed_dict)

                else:
                    # Test, no updates sent to parameter server:
                    # fetches = [self.fast_train_op,]
                    # fetched = sess.run(fetches, feed_dict=feed_dict) + [None, None]
                    fetched = [None, None]
                    # self.log.warning('test rollout ok.')

                if wirte_model_summary:
                    model_summary = fetched[-2]

                else:
                    model_summary = None

                if is_train:
                    # Copy from parameter server:
                    sess.run(self.sync_pi)

                    # Collect next train trajectory rollout:
                    train_data = self.get_data(data_sample_config=train_data_config)
                    feed_dict = self.process_data(sess, train_data, is_train=is_train)
                    # self.log.warning('Train data ok.')

                else:
                    train_data = self.get_data(data_sample_config=train_data_config)

                # Write down summaries:
                self.process_summary(sess, train_data, model_summary)
                self.local_steps += 1
                roll_num += 1

                # self.log.warning(
                #     'is_train: {}, jaggy_guide_loss: {}'.
                #         format(is_train, sess.run(self.guided_loss, feed_dict))
                # )
        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)