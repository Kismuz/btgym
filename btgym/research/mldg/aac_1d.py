import tensorflow as tf
import numpy as np

from btgym.algorithms.utils import batch_stack, batch_gather
from btgym.research.mldg.aac_1 import AMLDG_1
from btgym.research.mldg.memory import LocalMemory


class AMLDG_1d(AMLDG_1):
    """
    AMLDG_1 + tunable g1 + t2d methods
    """

    def __init__(
            self,
            g1_lambda=1.0,
            num_train_updates=1,
            train_batch_size=64,
            name='AMLDG1d',
            **kwargs
         ):

        self.g1_lambda = g1_lambda
        self.train_batch_size = train_batch_size
        self.num_train_updates = num_train_updates
        self.episode_memory = LocalMemory()
        super().__init__(name=name, **kwargs)

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
                meta_g = g1 * self.g1_lambda + g2

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
        self.inc_step = self.global_step.assign_add(tf.shape(pi_prime.on_state_in['external'])[0])

        # Local fast optimisation op:
        self.fast_train_op = self.fast_optimizer.apply_gradients(train_grads_and_vars)

        # Global meta-optimisation op:
        self.meta_train_op = self.optimizer.apply_gradients(meta_grads_and_vars)

        self.log.debug('train_op defined')
        return self.fast_train_op, self.meta_train_op

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
                feed_dict = {}
                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                self.episode_memory.add_batch(
                    **self.half_process_data(sess, train_data, is_train=is_train, pi=self.local_network)
                )
                # self.log.warning('Train roll added to memory.')

                for i in range(self.num_train_updates):
                    feed_dict = self._get_main_feeder(
                        sess,
                        **self.episode_memory.sample(self.train_batch_size),
                        is_train=is_train,
                        pi=self.local_network,
                        pi_prime=self.local_network_prime)

                    # self.log.warning('Train feed dict ok.')

                    fetches = [self.fast_train_op]

                    fetched = sess.run(fetches, feed_dict=feed_dict)

                    # self.log.warning('Pi_prime update {} ok.'.format(i))

                # Collect test rollout using [updated] pi_prime policy:
                test_data = self.get_data(
                    policy=self.local_network_prime,
                    data_sample_config=data_config
                )

                # self.log.warning('test_rollout_ok')

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
                # sess.run(self.sync_pi_from_prime)

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
