import tensorflow as tf
import numpy as np

import sys
from logbook import Logger, StreamHandler
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner

from btgym.algorithms.nn.layers import noisy_linear


class SubAAC(GuidedAAC):
    """
    Sub AAC trainer as lower-level part of meta-optimisation algorithm.
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


class AMLDG():
    """
    Asynchronous implementation of MLDG algorithm (by Da Li et al.)
    for one-shot adaptation in dynamically changing environments.

    This class is AAC wrapper; relies on sub-AAC classes to make separate policy networks
    for train/test data streams, performs data streams synchronization according to algorithm logic
    via data_config dictionaries; performs actual data checks to prevent test information leakage.

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
            opt_decay_steps=None,
            opt_end_learn_rate=None,
            opt_learn_rate=1e-4,
            fast_opt_learn_rate=1e-3,
            opt_max_env_steps=10 ** 7,
            aac_lambda=1.0,
            guided_lambda=1.0,
            rollout_length=20,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,  # one-shot adaptation
            _aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='AMLDG',
            **kwargs
    ):
        try:
            self.aac_class_ref = aac_class_ref
            self.task = task
            self.name = name
            self.summary_writer = None

            self.opt_learn_rate = opt_learn_rate
            self.opt_max_env_steps = opt_max_env_steps
            self.fast_opt_learn_rate = fast_opt_learn_rate

            if opt_end_learn_rate is None:
                self.opt_end_learn_rate = self.opt_learn_rate
            else:
                self.opt_end_learn_rate = opt_end_learn_rate

            if opt_decay_steps is None:
                self.opt_decay_steps = self.opt_max_env_steps
            else:
                self.opt_decay_steps = opt_decay_steps

            StreamHandler(sys.stdout).push_application()
            self.log = Logger('{}_{}'.format(name, task), level=log_level)
            self.rollout_length = rollout_length

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

            # Instantiate two sub-trainers: one for meta-test and one for meta-train environments:

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 1}  # master
            self.runner_config['kwargs']['name'] = 'master'

            self.train_aac = aac_class_ref(
                env=self.env_list[0],  # train data will be master environment
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                opt_learn_rate=self.fast_opt_learn_rate,  # non-decaying, used for fast pi_prime adaptation
                opt_max_env_steps=self.opt_max_env_steps,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                rollout_length=self.rollout_length,
                trial_source_target_cycle=trial_source_target_cycle,
                num_episodes_per_trial=num_episodes_per_trial,
                _use_target_policy=False,
                _use_global_network=True,
                _aux_render_modes=_aux_render_modes,
                name=self.name + '/metaTrain',
                **kwargs
            )

            self.runner_config['kwargs']['data_sample_config'] = {'mode': 0}  # slave
            self.runner_config['kwargs']['name'] = 'slave'

            self.test_aac = aac_class_ref(
                env=self.env_list[-1],  # test data -> slave env.
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                opt_learn_rate=0.0,  # test_aac.optimizer is not used
                opt_max_env_steps=self.opt_max_env_steps,
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
                name=self.name + '/metaTest',
                **kwargs
            )

            self.local_steps = self.train_aac.local_steps
            self.model_summary_freq = self.train_aac.model_summary_freq

            self._make_train_op()

            self.test_aac.model_summary_op = tf.summary.merge(
                [self.test_aac.model_summary_op, self._combine_meta_summaries()],
                name='meta_model_summary'
            )

        except:
            msg = 'AMLDG.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_train_op(self):
        """
        Defines tensors holding training op graph for meta-train, meta-test and meta-optimisation.
        """
        # Handy aliases:
        pi = self.train_aac.local_network  # local meta-train policy
        pi_prime = self.test_aac.local_network  # local meta-test policy
        pi_global = self.train_aac.network  # global shared policy

        self.test_aac.sync = self.test_aac.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
        )
        self.test_aac.sync_pi_global = self.test_aac.sync_global = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi_global.var_list)]
        )
        self.train_aac.sync_pi_local = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_prime.var_list)]
        )

        # Shared counters:
        self.global_step = self.train_aac.global_step
        self.global_episode = self.train_aac.global_episode

        self.test_aac.global_step = self.train_aac.global_step
        self.test_aac.global_episode = self.train_aac.global_episode
        self.test_aac.inc_episode = self.train_aac.inc_episode
        self.train_aac.inc_episode = None

        # Meta-opt. loss:
        self.loss = self.train_aac.loss + self.test_aac.loss

        # Clipped gradients:
        self.train_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.train_aac.loss, pi.var_list),
            40.0
        )
        self.test_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.test_aac.loss, pi_prime.var_list),
            40.0
        )
        # Aliases:
        pi.grads = self.train_aac.grads
        pi_prime.grads = self.test_aac.grads

        # Meta_optimisation gradients as an average of meta-train and meta-test gradients:
        self.grads = []
        for g1, g2 in zip(pi.grads, pi_prime.grads):
            if g1 is not None and g2 is not None:
                meta_g = (g1 + g2) / 2.0

            else:
                meta_g = None  # need to map correctly to vars

            self.grads.append(meta_g)

        # Gradients to update local meta-test policy (from train data):
        train_grads_and_vars = list(zip(pi.grads, pi_prime.var_list))

        # self.log.warning('train_grads_and_vars_len: {}'.format(len(train_grads_and_vars)))

        # Meta-gradients to be sent to parameter server:
        meta_grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # Remove empty entries:
        meta_grads_and_vars = [(g, v) for (g, v) in meta_grads_and_vars if g is not None]

        # for item in meta_grads_and_vars:
        #     self.log.warning('\nmeta_g_v: {}'.format(item))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.train_aac.local_network.on_state_in.keys())
        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.train_aac.inc_step = self.train_aac.global_step.assign_add(
            tf.shape(self.test_aac.local_network.on_state_in['external'])[0]
        )
        self.inc_step = self.train_aac.inc_step
        # Pi to pi_prime local adaptation op:
        # self.train_op = self.train_aac.optimizer.apply_gradients(train_grads_and_vars)

        # self.fast_opt = tf.train.GradientDescentOptimizer(self.alpha_rate)
        self.fast_opt = tf.train.GradientDescentOptimizer(self.fast_opt_learn_rate)
        self.train_op = self.fast_opt.apply_gradients(train_grads_and_vars)

        #  Learning rate annealing:
        self.learn_rate_decayed = tf.train.polynomial_decay(
            self.opt_learn_rate,
            self.global_step + 1,
            self.opt_decay_steps,
            self.opt_end_learn_rate,
            power=1,
            cycle=False,
        )

        # Optimizer for meta-update, sharing same learn rate (change?):
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate_decayed, epsilon=1e-5)

        # Global meta-optimisation op:
        self.meta_train_op = self.optimizer.apply_gradients(meta_grads_and_vars)

        self.log.debug('meta_train_op defined')

    def _combine_meta_summaries(self):
        """
        Additional summaries here.
        """
        meta_model_summaries = [
            tf.summary.scalar('meta_grad_global_norm', tf.global_norm(self.grads)),
            tf.summary.scalar('total_meta_loss', self.loss),
            #tf.summary.scalar('alpha_learn_rate', self.alpha_rate),
            #tf.summary.scalar('alpha_learn_rate_loss', self.alpha_rate_loss)
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

    def assert_trial_type(self, data, type):
        """
        Prevent information leakage:
        check actual trial type consistency; if failed - possible data sampling logic fault, issue warning.

        Args:
            data:
            type:   bool

        Returns:

        """
        try:
            assert (np.asarray(data['on_policy'][0]['state']['metadata']['trial_type']) == type).all()
        except AssertionError:
            self.log.warning(
                'Source trial assertion failed!\nExpected: `trial_type`={}\nGot metadata: {}'.\
                    format(type, data['on_policy'][0]['state']['metadata'])
            )

    def assert_same_trial(self, train_data, test_data):
        """
        Prevent information leakage-II:
        check if both data streams come from same trial.

        Args:
            train_data:
            test_data:
        """
        train_trial_chksum = np.average(train_data['on_policy'][0]['state']['metadata']['trial_num'])
        test_trial_chksum = np.average(test_data['on_policy'][0]['state']['metadata']['trial_num'])
        try:
            assert train_trial_chksum == test_trial_chksum
        except AssertionError:
            msg1 = 'Trials match assertion failed!\nGot train metadata: {},\nGot test metadata:  {}'. \
                format(
                train_data['on_policy'][0]['state']['metadata'],
                test_data['on_policy'][0]['state']['metadata']
            )
            msg2 = '\nTrain_trial_chksum: {}, test_trial_chksum: {}'.format(train_trial_chksum, test_trial_chksum)
            self.log.warning(msg1 + msg2)

    def assert_episode_type(self, data, type):
        """
        Prevent information leakage-III:
        check episode type for consistency; if failed - possible data sampling logic fault, issue warning.

        Args:
            train_data:
            test_data:
        """
        try:
            assert (np.asarray(data['on_policy'][0]['state']['metadata']['type']) == type).all()

        except AssertionError:
            msg = 'Episode types assertion failed!\nExpected episode_type: {},\nGot episode metadata:  {}'.\
                format(
                    type,
                    data['on_policy'][0]['state']['metadata']
            )
            self.log.warning(msg)

    def process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server:
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)
            # self.log.warning('Init Sync ok.')

            # Get data configuration,
            # (want both data streams come from  same trial,
            # and trial type we got can be either from source or target domain);
            # note: data_config counters get updated once per .process() call
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., samples trial
            test_data_config = self.train_aac.get_sample_config(mode=0)   # slave env, catches up with same trial

            # self.log.warning('train_data_config: {}'.format(train_data_config))
            # self.log.warning('test_data_config: {}'.format(test_data_config))

            # If this step data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_target = train_data_config['trial_config']['sample_type']
            done = False

            # Collect initial meta-train trajectory rollout:
            train_data = self.train_aac.get_data(data_sample_config=train_data_config, force_new_episode=True)
            feed_dict = self.train_aac.process_data(sess, train_data, is_train=True,pi=self.train_aac.local_network)

            # self.log.warning('Init Train data ok.')

            # Disable possibility of master data runner acquiring new trials,
            # in case meta-train episode termintaes earlier than meta-test -
            # we than need to get additional meta-train trajectories from exactly same distribution (trial):
            train_data_config['trial_config']['get_new'] = 0

            roll_num = 0

            # Collect entire meta-test episode rollout by rollout:
            while not done:
                # self.log.warning('Roll #{}'.format(roll_num))

                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                # self.log.warning(
                #     'Train data trial_num: {}'.format(
                #         np.asarray(train_data['on_policy'][0]['state']['metadata']['trial_num'])
                #     )
                # )

                # Paranoid checks against data sampling logic faults to prevent possible cheating:
                train_trial_chksum = np.average(train_data['on_policy'][0]['state']['metadata']['trial_num'])

                # Update pi_prime parameters wrt collected train data:
                if wirte_model_summary:
                    fetches = [self.train_op, self.train_aac.model_summary_op]
                else:
                    fetches = [self.train_op]

                fetched = sess.run(fetches, feed_dict=feed_dict)

                # self.log.warning('Train gradients ok.')

                # Collect test rollout using updated pi_prime policy:
                test_data = self.test_aac.get_data(data_sample_config=test_data_config)

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                # self.log.warning(
                #     'Test data trial_num: {}'.format(
                #         np.asarray(test_data['on_policy'][0]['state']['metadata']['trial_num'])
                #     )
                # )

                test_trial_chksum = np.average(test_data['on_policy'][0]['state']['metadata']['trial_num'])

                # Ensure slave runner data consistency, can correct if episode just started:
                if roll_num == 0 and train_trial_chksum != test_trial_chksum:
                    test_data = self.test_aac.get_data(data_sample_config=test_data_config, force_new_episode=True)
                    done = np.asarray(test_data['terminal']).any()
                    faulty_chksum = test_trial_chksum
                    test_trial_chksum = np.average(test_data['on_policy'][0]['state']['metadata']['trial_num'])

                    self.log.warning(
                        'Test trial corrected: {} -> {}'.format(faulty_chksum, test_trial_chksum)
                    )

                # self.log.warning(
                #     'roll # {}: train_trial_chksum: {}, test_trial_chksum: {}'.
                #         format(roll_num, train_trial_chksum, test_trial_chksum)
                # )

                if train_trial_chksum != test_trial_chksum:
                    # Still got error? - highly probable algorithm logic fault. Issue warning.
                    msg = 'Train/test trials mismatch found!\nGot train trials: {},\nTest trials: {}'. \
                        format(
                        train_data['on_policy'][0]['state']['metadata']['trial_num'][0],
                        test_data['on_policy'][0]['state']['metadata']['trial_num'][0]
                        )
                    msg2 = 'Train data config: {}\n Test data config: {}'.format(train_data_config, test_data_config)

                    self.log.warning(msg)
                    self.log.warning(msg2)

                # Check episode type for consistency; if failed - another data sampling logic fault, warn:
                try:
                    assert (np.asarray(test_data['on_policy'][0]['state']['metadata']['type']) == 1).all()
                    assert (np.asarray(train_data['on_policy'][0]['state']['metadata']['type']) == 0).all()
                except AssertionError:
                    msg = 'Train/test episodes types mismatch found!\nGot train ep. type: {},\nTest ep.type: {}'. \
                        format(
                        train_data['on_policy'][0]['state']['metadata']['type'],
                        test_data['on_policy'][0]['state']['metadata']['type']
                    )
                    self.log.warning(msg)

                # self.log.warning('Test data ok.')

                if not is_target:
                    # Process test data and perform meta-optimisation step:
                    feed_dict.update(
                        self.test_aac.process_data(sess, test_data, is_train=True, pi=self.test_aac.local_network)
                    )

                    if wirte_model_summary:
                        meta_fetches = [self.meta_train_op, self.test_aac.model_summary_op, self.inc_step]
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
                # copy from parameter server, not while testing:
                if not is_target:
                    sess.run(self.train_aac.sync_pi)

                # Copy from pi to pi-prime:
                sess.run(self.test_aac.sync_pi)
                # self.log.warning('Sync ok.')

                # Collect next train trajectory rollout:
                train_data = self.train_aac.get_data(data_sample_config=train_data_config)
                feed_dict = self.train_aac.process_data(sess,train_data, is_train=True, pi=self.train_aac.local_network)
                # self.log.warning('Train data ok.')

                # Write down summaries:
                self.test_aac.process_summary(sess, test_data, meta_model_summary)
                self.train_aac.process_summary(sess, train_data, model_summary)
                self.local_steps += 1
                roll_num += 1
        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


class AMLDG_2(AMLDG):
    """
    FAILED do not use
    """

    def __init__(self, name='AMLDGv2', **kwargs):
        super(AMLDG_2, self).__init__(name=name, **kwargs)

    def process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server:
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)
            # self.log.warning('Init Sync ok.')

            # Get data configuration,
            #  Want data streams come from  different trials from same doman!
            # note: data_config counters get updated once per .process() call
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., samples trial
            test_data_config = self.train_aac.get_sample_config(mode=0)  # slave env, catches up with same trial

            # self.log.warning('train_data_config: {}'.format(train_data_config))
            # self.log.warning('test_data_config: {}'.format(test_data_config))

            # If this step data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_target = train_data_config['trial_config']['sample_type']
            done = False

            # Collect initial meta-train trajectory rollout:
            train_data = self.train_aac.get_data(data_sample_config=train_data_config, force_new_episode=True)
            feed_dict = self.train_aac.process_data(sess, train_data,is_train=True, pi=self.train_aac.local_network)

            # self.log.warning('Init Train data ok.')

            # For target domain only:
            # Disable possibility of master data runner acquiring new trials,
            # in case meta-train episode termintaes earlier than meta-test -
            # we than need to get additional meta-train trajectories from exactly same distribution (trial):
            if is_target:
                train_data_config['trial_config']['get_new'] = 0

            roll_num = 0

            # Collect entire meta-test episode rollout by rollout:
            while not done:
                # self.log.warning('Roll #{}'.format(roll_num))

                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                # if not is_target:
                # Update pi_prime parameters wrt collected train data:
                if wirte_model_summary:
                    fetches = [self.train_op, self.train_aac.model_summary_op]
                else:
                    fetches = [self.train_op]

                fetched = sess.run(fetches, feed_dict=feed_dict)
                # else:
                #     # Target domain test, no local policy update:
                #     fetched = [None, None]

                # self.log.warning('Train gradients ok.')

                # Collect test rollout using [updated] pi_prime policy:
                test_data = self.test_aac.get_data(data_sample_config=test_data_config)

                # self.log.warning('Test_data:')
                # for k, v in test_data.items():
                #     self.log.warning(
                #         'Key: {}, value_type: {},  value_shape: {}'.format(k, type(v), np.asarray(v).shape)
                #     )

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                # # Reset master env to new trial to decorellate: TODO: quick fix, this one roll gets waisted, change
                # if roll_num == 0:
                #     train_data = self.train_aac.get_data(data_sample_config=train_data_config, force_new_episode=True)
                #     self.assert_trial_type(train_data, is_target)
                self.assert_trial_type(train_data, is_target)
                self.assert_trial_type(test_data, is_target)

                self.assert_episode_type(train_data, 0)
                self.assert_episode_type(test_data, 1)

                # self.assert_same_trial(train_data, test_data)

                # self.log.warning('Test data ok.')

                if not is_target:
                    # Process test data and perform meta-optimisation step:
                    feed_dict.update(
                        self.test_aac.process_data(sess, test_data,is_train=True, pi=self.test_aac.local_network)
                    )

                    if wirte_model_summary:
                        meta_fetches = [self.meta_train_op, self.test_aac.model_summary_op, self.inc_step]
                    else:
                        meta_fetches = [self.meta_train_op, self.inc_step]

                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                    # self.log.warning('Meta-gradients ok.')
                else:
                    self.assert_same_trial(train_data, test_data)
                    # Target domain test, no updates sent to parameter server:
                    meta_fetched = [None, None]

                    # self.log.warning('Meta-opt. rollout ok.')

                if wirte_model_summary:
                    meta_model_summary = meta_fetched[-2]
                    model_summary = fetched[-1]

                else:
                    meta_model_summary = None
                    model_summary = None

                # Next step housekeeping:
                # copy from parameter server:
                sess.run(self.train_aac.sync_pi)
                sess.run(self.test_aac.sync_pi)  # TODO: maybe not?
                # self.log.warning('Sync ok.')

                # if not is_target:
                # Collect next train trajectory rollout:
                train_data = self.train_aac.get_data(data_sample_config=train_data_config)

                # Concatenate new train and previous step test data:
                joined_data = {
                    k: train_data[k] + test_data[k] for k in ['on_policy', 'terminal', 'off_policy', 'off_policy_rp']
                }
                train_data.update(joined_data)

                # self.log.warning('Train_data:')
                # for k, v in train_data.items():
                #     self.log.warning(
                #         'Key: {}, value_type: {},  value_shape: {}'.format(k, type(v), np.asarray(v).shape)
                #     )

                feed_dict = self.train_aac.process_data(sess, train_data,is_train=True, pi=self.train_aac.local_network)

                # self.log.warning('Train data ok.')
                self.train_aac.process_summary(sess, train_data, model_summary)

                # test summary anyway:
                self.test_aac.process_summary(sess, test_data, meta_model_summary)

                self.local_steps += 1
                roll_num += 1
        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


class AMLDG_3(AMLDG):
    """
    FAILED do not use
    Closed-loop meta-update.
    """

    def __init__(self, fast_learn_rate_train=0.1, fast_learn_rate_test=0.1, name='AMLDGv3', **kwargs):
        self.fast_learn_rate_train = fast_learn_rate_train
        self.fast_learn_rate_test = fast_learn_rate_test
        super(AMLDG_3, self).__init__(name=name, **kwargs)

    def _make_train_op(self):
        """
        Defines tensors holding training op graph for meta-train, meta-test and meta-optimisation.
        """
        # Handy aliases:
        pi = self.train_aac.local_network  # local meta-train policy
        pi_prime = self.test_aac.local_network  # local meta-test policy
        pi_global = self.train_aac.network  # global shared policy

        self.test_aac.sync = self.test_aac.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
        )
        self.train_aac.sync_pi_local = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_prime.var_list)]
        )

        # Shared counters:
        self.global_step = self.train_aac.global_step
        self.global_episode = self.train_aac.global_episode

        self.test_aac.global_step = self.train_aac.global_step
        self.test_aac.global_episode = self.train_aac.global_episode
        self.test_aac.inc_episode = self.train_aac.inc_episode
        self.train_aac.inc_episode = None
        self.inc_step = self.train_aac.inc_step

        # Meta-opt. loss:
        self.loss = self.train_aac.loss + self.test_aac.loss

        # Clipped gradients:
        self.train_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.train_aac.loss, pi.var_list),
            40.0
        )
        self.test_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.test_aac.loss, pi_prime.var_list),
            40.0
        )
        # Aliases:
        pi.grads = self.train_aac.grads
        pi_prime.grads = self.test_aac.grads

        # Meta_optimisation gradients as sum of meta-train and meta-test gradients:
        self.grads = []
        for g1, g2 in zip(pi.grads, pi_prime.grads):
            if g1 is not None and g2 is not None:
                meta_g = g1 + g2
                # meta_g = (1 - self.meta_grads_scale) * g1 + self.meta_grads_scale * g2
            else:
                meta_g = None  # need to map correctly to vars

            self.grads.append(meta_g)

        # Gradients to update local meta-test policy (from train data):
        train_grads_and_vars = list(zip(pi.grads, pi_prime.var_list))

        # self.log.warning('train_grads_and_vars_len: {}'.format(len(train_grads_and_vars)))

        # Meta-gradients to be sent to parameter server:
        meta_grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # Remove empty entries:
        meta_grads_and_vars = [(g, v) for (g, v) in meta_grads_and_vars if g is not None]

        # for item in meta_grads_and_vars:
        #     self.log.warning('\nmeta_g_v: {}'.format(item))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(self.train_aac.local_network.on_state_in.keys())
        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        self.train_aac.inc_step = self.train_aac.global_step.assign_add(
            tf.shape(self.train_aac.local_network.on_state_in['external'])[0]
        )
        # Simple SGD, no average statisitics:
        self.fast_optimizer_train = tf.train.GradientDescentOptimizer(self.fast_learn_rate_train)
        self.fast_optimizer_test = tf.train.GradientDescentOptimizer(self.fast_learn_rate_test)

        # Pi to pi_prime local adaptation op:
        self.train_op = self.fast_optimizer_train.apply_gradients(train_grads_and_vars)

        # Optimizer for meta-update, sharing same learn rate (change?):
        self.optimizer = tf.train.AdamOptimizer(self.train_aac.train_learn_rate, epsilon=1e-5)

        # Global meta-optimisation op:
        self.meta_train_op = self.optimizer.apply_gradients(meta_grads_and_vars)

        # Local meta-optimisation:
        local_meta_grads_and_vars = list(zip(self.grads, pi_prime.var_list))
        local_meta_grads_and_vars = [(g, v) for (g, v) in local_meta_grads_and_vars if g is not None]

        self.local_meta_train_op = self.fast_optimizer_test.apply_gradients(local_meta_grads_and_vars)

        self.log.debug('meta_train_op defined')

    def process(self, sess):
        """
        Meta-train/test procedure for one-shot learning.
        Single call runs single meta-test episode.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server into both policies:
            sess.run(self.train_aac.sync_pi)
            sess.run(self.test_aac.sync_pi)
            # self.log.warning('Init Sync ok.')

            # Get data configuration,
            # (want both data streams come from  same trial,
            # and trial type we got can be either from source or target domain);
            # note: data_config counters get updated once per .process() call
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., samples trial
            test_data_config = self.train_aac.get_sample_config(mode=0)   # slave env, catches up with same trial

            # self.log.warning('train_data_config: {}'.format(train_data_config))
            # self.log.warning('test_data_config: {}'.format(test_data_config))

            # If this episode data comes from source or target domain
            # (i.e. is it either meta-optimised or true test episode):
            is_target = train_data_config['trial_config']['sample_type']
            done = False

            # Collect initial meta-train trajectory rollout:
            train_data = self.train_aac.get_data(data_sample_config=train_data_config, force_new_episode=True)
            feed_dict =self.train_aac.process_data(sess, train_data,is_train=True, pi=self.train_aac.local_network)

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

                # Paranoid checks against data sampling logic faults to prevent possible cheating:
                train_trial_chksum = np.average(train_data['on_policy'][0]['state']['metadata']['trial_num'])

                # Update pi_prime parameters wrt collected train data:
                if wirte_model_summary:
                    fetches = [self.train_op, self.train_aac.model_summary_op]
                else:
                    fetches = [self.train_op]

                fetched = sess.run(fetches, feed_dict=feed_dict)

                # self.log.warning('Train gradients ok.')

                # Collect test rollout using updated pi_prime policy:
                test_data = self.test_aac.get_data(data_sample_config=test_data_config)

                # If meta-test episode has just ended?
                done = np.asarray(test_data['terminal']).any()

                test_trial_chksum = np.average(test_data['on_policy'][0]['state']['metadata']['trial_num'])

                # Ensure slave runner data consistency, can correct if episode just started:
                if roll_num == 0 and train_trial_chksum != test_trial_chksum:
                    test_data = self.test_aac.get_data(data_sample_config=test_data_config, force_new_episode=True)
                    done = np.asarray(test_data['terminal']).any()
                    faulty_chksum = test_trial_chksum
                    test_trial_chksum = np.average(test_data['on_policy'][0]['state']['metadata']['trial_num'])

                    self.log.warning(
                        'Test trial corrected: {} -> {}'.format(faulty_chksum, test_trial_chksum)
                    )

                if train_trial_chksum != test_trial_chksum:
                    # Still got error? - highly probable algorithm logic fault. Issue warning.
                    msg = 'Train/test trials mismatch found!\nGot train trials: {},\nTest trials: {}'. \
                        format(
                        train_data['on_policy'][0]['state']['metadata']['trial_num'][0],
                        test_data['on_policy'][0]['state']['metadata']['trial_num'][0]
                        )
                    msg2 = 'Train data config: {}\n Test data config: {}'.format(train_data_config, test_data_config)

                    self.log.warning(msg)
                    self.log.warning(msg2)

                # Check episode type for consistency; if failed - another data sampling logic fault, warn:
                try:
                    assert (np.asarray(test_data['on_policy'][0]['state']['metadata']['type']) == 1).any()
                    assert (np.asarray(train_data['on_policy'][0]['state']['metadata']['type']) == 0).any()
                except AssertionError:
                    msg = 'Train/test episodes types mismatch found!\nGot train ep. type: {},\nTest ep.type: {}'. \
                        format(
                        train_data['on_policy'][0]['state']['metadata']['type'],
                        test_data['on_policy'][0]['state']['metadata']['type']
                    )
                    self.log.warning(msg)

                # self.log.warning('Test data ok.')

                # Process test data and perform meta-optimisation step:
                feed_dict.update(
                    self.test_aac.process_data(sess, test_data, is_train=True, pi=self.test_aac.local_network)
                )

                if not is_target:
                    # Update local pi_prime (with fast_learn_rate) and global shared parameters (via slow_learn_rate):
                    if wirte_model_summary:
                        meta_fetches = [
                            self.meta_train_op,
                            self.local_meta_train_op,
                            self.test_aac.model_summary_op,
                            self.inc_step
                        ]
                    else:
                        # Only update local pi_prime:
                        meta_fetches = [
                            self.meta_train_op,
                            self.local_meta_train_op,
                            self.inc_step
                        ]

                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                    # self.log.warning('Meta-gradients ok.')
                else:
                    # True test, no updates sent to parameter server:
                    meta_fetches = [self.local_meta_train_op]
                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict) + [None, None]
                    # self.log.warning('Meta-opt. rollout ok.')

                if wirte_model_summary:
                    meta_model_summary = meta_fetched[-2]
                    model_summary = fetched[-1]

                else:
                    meta_model_summary = None
                    model_summary = None

                # Copy pi_prime to pi:
                sess.run(self.train_aac.sync_pi_local)
                # sess.run(self.test_aac.sync_pi)
                # self.log.warning('Sync ok.')

                # Collect next train trajectory rollout:
                train_data = self.train_aac.get_data(data_sample_config=train_data_config)
                feed_dict = self.train_aac.process_data(sess, train_data,is_train=True, pi=self.train_aac.local_network)
                # self.log.warning('Train data ok.')

                # Write down summaries:
                self.test_aac.process_summary(sess, test_data, meta_model_summary)
                self.train_aac.process_summary(sess, train_data, model_summary)
                self.local_steps += 1
                roll_num += 1
        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)
