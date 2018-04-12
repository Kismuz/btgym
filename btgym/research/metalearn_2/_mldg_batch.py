import tensorflow as tf
import numpy as np

import sys
from logbook import Logger, StreamHandler

from btgym.research.mldg.aac import SubAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class MLDG():
    """
    Asynchronous implementation of MLDG algorithm
    for continuous adaptation in dynamically changing environments.

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
            env,
            task,
            log_level,
            aac_class_ref=SubAAC,
            runner_config=None,
            aac_lambda=1.0,
            guided_lambda=1.0,
            rollout_length=20,
            train_support=300,
            fast_adapt_num_steps=10,
            fast_adapt_batch_size=32,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='MLDG',
            **kwargs
    ):
        try:
            self.aac_class_ref = aac_class_ref
            self.task = task
            self.name = name
            self.summary_writer = None

            StreamHandler(sys.stdout).push_application()
            self.log = Logger('{}_{}'.format(name, task), level=log_level)

            self.rollout_length = rollout_length
            self.train_support = train_support  # number of train experiences to collect
            self.train_batch_size = int(self.train_support / self.rollout_length)
            self.fast_adapt_num_steps = fast_adapt_num_steps
            self.fast_adapt_batch_size = fast_adapt_batch_size

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

            # Instantiate two sub-trainers: one for test and one for train environments:

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 1}  # master
            self.runner_config['kwargs']['name'] = 'master'

            self.train_aac = aac_class_ref(
                env=self.env_list[0],  # train data will be master environment TODO: really dumb data control. improve.
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                rollout_length=self.rollout_length,
                trial_source_target_cycle=trial_source_target_cycle,
                num_episodes_per_trial=num_episodes_per_trial,
                _use_target_policy=False,
                _use_global_network=True,
                _aux_render_modes=_aux_render_modes,
                name=self.name + '_sub_Train',
                **kwargs
            )

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 0}  # master
            self.runner_config['kwargs']['name'] = 'slave'

            self.test_aac = aac_class_ref(
                env=self.env_list[-1],  # test data -> slave env.
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                rollout_length=self.rollout_length,
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
            msg = 'MLDG.__init()__ exception occurred' + \
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

        # self.log.warning('train_grads_and_vars_len: {}'.format(len(train_grads_and_vars)))

        # Meta-gradients to be sent to parameter server:
        meta_grads_and_vars = list(zip(self.grads, self.train_aac.network.var_list))

        # self.log.warning('meta_grads_and_vars_len: {}'.format(len(meta_grads_and_vars)))

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
            self.train_aac._start_runners(   # master first
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.train_aac.get_sample_config(mode=1)
            )
            self.test_aac._start_runners(
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.test_aac.get_sample_config(mode=0)
            )

            self.summary_writer = summary_writer
            self.log.notice('Runners started.')

        except:
            msg = 'start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def fast_adapt_step(self, sess, batch_size, on_policy_batch, off_policy_batch, rp_batch, make_summary=False):
        """
        One step of test_policy adaptation.

        Args:
            sess:                   tensorflow.Session obj.
            batch_size:             train mini-batch size
            on_policy_batch:        `on_policy` train data
            off_policy_batch:       `off_policy` train data or None
            rp_batch:               'reward_prediction` train data or None
            make_summary:           bool, if True - compute model summary

        Returns:
            model summary or None
        """
        # Sample from train distribution:
        on_mini_batch = self.train_aac.sample_batch(on_policy_batch, batch_size)
        off_mini_batch = self.train_aac.sample_batch(off_policy_batch, batch_size)
        rp_mini_batch = self.train_aac.sample_batch(rp_batch, batch_size)

        feed_dict = self.train_aac._get_main_feeder(sess, on_mini_batch, off_mini_batch, rp_mini_batch, True)

        if make_summary:
            fetches = [self.train_op, self.train_aac.model_summary_op]
        else:
            fetches = [self.train_op]

        # Update pi_prime parameters wrt sampled data:
        fetched = sess.run(fetches, feed_dict=feed_dict)

        # self.log.warning('Train gradients ok.')

        if make_summary:
            summary =  fetched[-1]

        else:
            summary = None

        return summary

    def train_step(self, sess, data_config):
        """
        Collects train task data and updates test policy parameters (fast adaptation).

        Args:
            sess:                   tensorflow.Session obj.
            data_config:            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`

        Returns:
            batched train data

        """
        # Collect train distribution:
        train_batch = self.train_aac.get_batch(
            size=self.train_batch_size,
            require_terminal=True,
            same_trial=True,
            data_sample_config=data_config
        )

        # for rollout in train_batch['on_policy']:
        #     self.log.warning(
        #         'Train data trial_num: {}'.format(
        #             np.asarray(rollout['state']['metadata']['trial_num'])
        #         )
        #     )

        # Process time-flat-alike (~iid) to treat as empirical data distribution over train task:
        on_policy_batch, off_policy_batch, rp_batch = self.train_aac.process_batch(sess, train_batch)

        # self.log.warning('Train data ok.')

        local_step = sess.run(self.global_step)
        local_episode = sess.run(self.global_episode)
        model_summary = None

        # Extract all non-empty summaries:
        ep_summary = [summary for summary in train_batch['ep_summary'] if summary is not None]

        # Perform number of test policy updates wrt. collected train data:
        for i in range(self.fast_adapt_num_steps):
            model_summary = self.fast_adapt_step(
                sess,
                batch_size=self.fast_adapt_batch_size,
                on_policy_batch=on_policy_batch,
                off_policy_batch=off_policy_batch,
                rp_batch=rp_batch,
                make_summary=(local_step + i) % self.model_summary_freq == 0
            )
            # self.log.warning('Batch {} Train gradients ok.'.format(i))

            # Write down summaries:
            train_summary = dict(
                render_summary=[None],
                test_ep_summary=[None],
                ep_summary=[ep_summary.pop() if len(ep_summary) > 0 else None]
            )
            self.train_aac.process_summary(
                sess,
                train_summary,
                model_summary,
                step=local_step + i,
                episode=local_episode + i
            )

        return on_policy_batch, off_policy_batch, rp_batch

    def meta_train_step(self, sess, data_config, on_policy_batch, off_policy_batch, rp_batch):
        """
        Collects data from source domain test task and performs meta-update to shared parameters vector.
        Writes down relevant summaries.

        Args:
            sess:                   tensorflow.Session obj.
            data_config:            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
            on_policy_batch:        `on_policy` train data
            off_policy_batch:       `off_policy` train data or None
            rp_batch:               'reward_prediction` train data or None

        """
        done = False
        while not done:
            # Say `No` to redundant summaries:
            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            # Collect test trajectory wrt updated test_policy parameters:
            test_data = self.test_aac.get_data(
                init_context=None,
                data_sample_config=data_config
            )
            test_batch_size = 0  # TODO: adjust on/off/rp sizes
            for rollout in test_data['on_policy']:
                test_batch_size += len(rollout['position'])

            test_feed_dict = self.test_aac.process_data(sess,,,,, test_data,,

                             # self.log.warning('Test data rollout for step {} ok.'.format(self.local_steps))
                             #
                             # self.log.warning(
                             #     'Test data trial_num: {}'.format(
                             #         np.asarray(test_data['on_policy'][0]['state']['metadata']['trial_num'])
                             #     )
                             # )

                             # Sample train data of same size:
                             feed_dict = self.train_aac._get_main_feeder(
                sess,
                self.train_aac.sample_batch(on_policy_batch, test_batch_size),
                self.train_aac.sample_batch(off_policy_batch, test_batch_size),
                self.train_aac.sample_batch(rp_batch, test_batch_size),
                True
            )
            # Add test trajectory:
            feed_dict.update(test_feed_dict)

            # Perform meta-update:
            if wirte_model_summary:
                meta_fetches = [self.meta_train_op, self.test_aac.model_summary_op, self.inc_step]
            else:
                meta_fetches = [self.meta_train_op, self.inc_step]

            meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

            # self.log.warning('Meta-gradients ok.')

            if wirte_model_summary:
                meta_model_summary = meta_fetched[-2]

            else:
                meta_model_summary = None

            # Write down summaries:
            self.test_aac.process_summary(sess, test_data, meta_model_summary)
            self.local_steps += 1

            # If test episode ended?
            done = np.asarray(test_data['terminal']).any()

    def meta_test_step(self, sess, data_config, on_policy_batch, off_policy_batch, rp_batch):
        """
        Validates adapted policy on data from target domain test task.
        Writes down relevant summaries.

        Args:
            sess:                   tensorflow.Session obj.
            data_config:            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
            on_policy_batch:        `on_policy` train data
            off_policy_batch:       `off_policy` train data or None
            rp_batch:               'reward_prediction` train data or None

        """
        done = False
        while not done:
            # Collect test trajectory:
            test_data = self.test_aac.get_data(
                init_context=None,
                data_sample_config=data_config
            )

            # self.log.warning('Target test rollout ok.')
            # self.log.warning(
            #     'Test data target trial_num: {}'.format(
            #         np.asarray(test_data['on_policy'][0]['state']['metadata']['trial_num'])
            #     )
            # )
            # self.log.warning('target_render_ep_summary: {}'.format(test_data['render_summary']))

            # Write down summaries:
            self.test_aac.process_summary(sess, test_data)

            # If test episode ended?
            done = np.asarray(test_data['terminal']).any()

    def process(self, sess):
        """
        Meta-train procedure for one-shot learning/

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server:
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)

            #self.log.warning('Sync ok.')

            # Decide on data configuration for train/test trajectories,
            # such as all data will come from same trial (maybe different episodes)
            # and trial type as well (~from source or target domain):
            # note: data_config counters get updated once per process() call
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., draws trial
            test_data_config = self.train_aac.get_sample_config(mode=0)   # slave env, catches up with same trial

            # If data comes from source or target domain:
            is_target = train_data_config['trial_config']['sample_type']

            # self.log.warning('PROCESS_train_data_config: {}'.format(train_data_config))
            # self.log.warning('PROCESS_test_data_config: {}'.format(test_data_config))

            # Fast adaptation step:
            # collect train trajectories, process time-flat-alike (~iid) to treat as empirical data distribution
            # over train task and adapt test_policy wrt. train experience:
            on_policy_batch, off_policy_batch, rp_batch = self.train_step(sess, train_data_config)

            # Slow adaptation step:
            if is_target:
                # Meta-test:
                # self.log.warning('Running meta-test episode...')
                self.meta_test_step(sess,test_data_config, on_policy_batch, off_policy_batch, rp_batch)

            else:
                # Meta-train:
                # self.log.warning('Running meta-train episode...')
                self.meta_train_step(sess,test_data_config, on_policy_batch, off_policy_batch, rp_batch)

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

