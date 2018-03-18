import tensorflow as tf
import numpy as np

import sys
from logbook import Logger, StreamHandler

from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner
from btgym.algorithms.memory import Memory


class SubAAC(GuidedAAC):
    """
    Sub AAC trainers as part of meta-trainer.
    """

    def __init__(
            self,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,
            **kwargs
    ):
        super(SubAAC, self).__init__(**kwargs)
        self.current_data = None
        self.current_feed_dict = None

        # Trials sampling control:
        self.num_source_trials = trial_source_target_cycle[0]
        self.num_target_trials = trial_source_target_cycle[-1]
        self.num_episodes_per_trial = num_episodes_per_trial

        # Note that only master (test runner) is requesting trials

        self.current_source_trial = 0
        self.current_target_trial = 0
        self.current_trial_mode = 0  # source
        self.current_episode = 0

    def process(self, sess):
        """
        self.process() logic is defined by meta-trainer.
        """
        pass

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


class MetaAAC_1_0():
    """
    Meta-trainer class.
    Implementation of `gradient-based meta-learning algorithm suitable
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
            env,
            task,
            log_level,
            aac_class_ref=SubAAC,
            runner_config=None,
            aac_lambda=1.0,
            guided_lambda=1.0,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='MetaAAC',
            **kwargs
    ):
        try:
            self.aac_class_ref = aac_class_ref
            self.task = task
            self.name = name
            StreamHandler(sys.stdout).push_application()
            self.log = Logger('{}_{}'.format(name, task), level=log_level)

            # with tf.variable_scope(self.name):
            if runner_config is None:
                self.runner_config = {
                    'class_ref': BaseSynchroRunner,
                    'kwargs': {},
                }
            else:
                self.runner_config = runner_config

            self.env_list = env

            assert isinstance(self.env_list, list) and len(self.env_list) == 2, \
                'Expected pair of environments, got: {}'.format(self.env_list)

            # Instantiate to sub-trainers: one for test and one for train environments:

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 0}  # salve
            self.runner_config['kwargs']['name'] = 'slave'

            self.train_aac = aac_class_ref(
                env=self.env_list[-1],  # train data will be salve environment
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                trial_source_target_cycle=trial_source_target_cycle,
                num_episodes_per_trial=num_episodes_per_trial,
                _use_target_policy=False,
                _use_global_network=True,
                _aux_render_modes=_aux_render_modes,
                name=self.name + '_sub_Train',
                **kwargs
            )

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 1}  # master
            self.runner_config['kwargs']['name'] = 'master'

            self.test_aac = aac_class_ref(
                env=self.env_list[0],  # test data - master env.
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                trial_source_target_cycle=trial_source_target_cycle,
                num_episodes_per_trial=num_episodes_per_trial,
                _use_target_policy=False,
                _use_global_network=False,
                global_step_op=self.train_aac.global_step,
                global_episode_op=self.train_aac.global_episode,
                inc_episode_op=self.train_aac.inc_episode,
                _aux_render_modes=_aux_render_modes,
                name=self.name + '_sub_Test',
                **kwargs
            )

            self.local_steps = self.train_aac.local_steps
            self.model_summary_freq = self.train_aac.model_summary_freq
            #self.model_summary_op = self.train_aac.model_summary_op

            self._make_train_op()
            self.test_aac.model_summary_op = tf.summary.merge(
                [self.test_aac.model_summary_op, self._combine_meta_summaries()],
                name='meta_model_summary'
            )

        except:
            msg = 'MetaAAC_0_1.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_train_op(self):
        """

        Defines:
            tensors holding training op graph for sub trainers and self;
        """
        pi = self.train_aac.local_network
        pi_prime = self.test_aac.local_network

        self.test_aac.sync = self.test_aac.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
        )

        self.global_step = self.train_aac.global_step
        self.global_episode = self.train_aac.global_episode

        self.test_aac.global_step = self.train_aac.global_step
        self.test_aac.global_episode = self.train_aac.global_episode
        self.test_aac.inc_episode = self.train_aac.inc_episode
        self.train_aac.inc_episode = None
        self.inc_step = self.train_aac.inc_step

        # Meta-loss:
        self.loss = 0.5 * self.train_aac.loss + 0.5 * self.test_aac.loss

        # Clipped gradients:
        self.train_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.train_aac.loss, pi.var_list),
            40.0
        )
        self.log.warning('self.train_aac.grads: {}'.format(len(list(self.train_aac.grads))))

        # self.test_aac.grads, _ = tf.clip_by_global_norm(
        #     tf.gradients(self.test_aac.loss, pi_prime.var_list),
        #     40.0
        # )
        # Meta-gradient:
        grads_i, _ = tf.clip_by_global_norm(
            tf.gradients(self.train_aac.loss, pi.var_list),
            40.0
        )

        grads_i_next, _ = tf.clip_by_global_norm(
            tf.gradients(self.test_aac.loss, pi_prime.var_list),
            40.0
        )

        self.grads = []
        for g1, g2 in zip(grads_i, grads_i_next):
            if g1 is not None and g2 is not None:
                meta_g = 0.5 * g1 + 0.5 * g2
            else:
                meta_g = None

            self.grads.append(meta_g)

        #self.log.warning('self.grads_len: {}'.format(len(list(self.grads))))

        # Gradients to update local copy of pi_prime (from train data):
        train_grads_and_vars = list(zip(self.train_aac.grads, pi_prime.var_list))

        self.log.warning('train_grads_and_vars_len: {}'.format(len(train_grads_and_vars)))

        # Meta-gradients to be sent to parameter server:
        meta_grads_and_vars = list(zip(self.grads, self.train_aac.network.var_list))

        self.log.warning('meta_grads_and_vars_len: {}'.format(len(meta_grads_and_vars)))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.train_aac.local_network.on_state_in.keys())

        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.train_aac.inc_step = self.train_aac.global_step.assign_add(
            tf.shape(self.train_aac.local_network.on_state_in['external'])[0]
        )

        self.train_op = self.train_aac.optimizer.apply_gradients(train_grads_and_vars)

        # Optimizer for meta-update:
        self.optimizer = tf.train.AdamOptimizer(self.train_aac.train_learn_rate, epsilon=1e-5)
        # TODO: own alpha-leran rate
        self.meta_train_op = self.optimizer.apply_gradients(meta_grads_and_vars)

        self.log.debug('meta_train_op defined')

    def _combine_meta_summaries(self):

        meta_model_summaries = [
            tf.summary.scalar("meta_grad_global_norm", tf.global_norm(self.grads)),
            tf.summary.scalar("total_meta_loss", self.loss),
        ]

        return meta_model_summaries

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
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)

            # Start thread_runners:
            self.test_aac._start_runners(sess, summary_writer)  # master first
            self.train_aac._start_runners(sess, summary_writer)

            self.summary_writer = summary_writer
            self.log.notice('Runners started.')

        except:
            msg = 'start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def process(self, sess):
        """
        Meta-train step.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Say `No` to redundant summaries:
            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            # Copy from parameter server:
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)

            #self.log.warning('Sync ok.')

            # Collect train trajectory:
            train_data = self.train_aac.get_data()
            feed_dict = self.train_aac.process_data(sess, train_data, is_train=True)

            #self.log.warning('Train data ok.')

            # Update pi_prime parameters wrt collected data:
            if wirte_model_summary:
                fetches = [self.train_op, self.train_aac.model_summary_op]
            else:
                fetches = [self.train_op]

            fetched = sess.run(fetches, feed_dict=feed_dict)

            #self.log.warning('Train gradients ok.')

            # Collect test trajectory wrt updated pi_prime parameters:
            test_data = self.test_aac.get_data()
            feed_dict.update(self.test_aac.process_data(sess, test_data, is_train=True))

            #self.log.warning('Test data ok.')

            # Perform meta-update:
            if wirte_model_summary:
                meta_fetches = [self.meta_train_op, self.test_aac.model_summary_op, self.inc_step]
            else:
                meta_fetches = [self.meta_train_op, self.inc_step]

            meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

            #self.log.warning('Meta-gradients ok.')

            if wirte_model_summary:
                meta_model_summary = meta_fetched[-2]
                model_summary = fetched[-1]

            else:
                meta_model_summary = None
                model_summary = None

            # Write down summaries:
            self.test_aac.process_summary(sess, test_data, meta_model_summary)
            self.train_aac.process_summary(sess, train_data, model_summary)
            self.local_steps += 1

            # TODO: ...what about sampling control?

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)




