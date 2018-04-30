import tensorflow as tf
import numpy as np

from btgym.algorithms.utils import batch_stack, batch_gather
from btgym.research.mldg.aac import AMLDG, SubAAC


class SubAAC_d(SubAAC):
    """
    SubAAC + methods to treat train trajectories as empirical distributions
    """
    def __init__(self, name='SubAACd', **kwargs):
        super(SubAAC_d, self).__init__(name=name, **kwargs)

    def get_batch(self, **kwargs):
        """
        Retrieves batch of rollouts from runners.

        Args:
            **kwargs:   see btgym.algorithms.runner.synchro.BaseSynchroRunner.get_batch() method args.

        Returns:
            dictionary of batched rollouts
        """
        rollouts = []
        terminal_context = []
        for runner in self.runners:
            batch = runner.get_batch(**kwargs)
            for rollout in batch['data']:
                rollouts.append(rollout)

            for context in batch['terminal_context']:
                terminal_context.append(context)

        self.log.debug('rollouts_len: {}'.format(len(rollouts)))

        final_batch = {key: [rollout[key] for rollout in rollouts] for key in rollouts[0].keys()}
        final_batch['terminal_context'] = terminal_context

        return final_batch

    @staticmethod
    def sample_batch(batch, sample_size):
        """
        Uniformly randomly samples mini-batch from (supposedly bigger) batch.

        Args:
            batch:          nested dict of experiences
            sample_size:    mini-batch size

        Returns:
            nested dict of experiences of same structure as `batch` with number of experiences eq. to `sample_size`.
        """
        if batch is not None:
            batch_size = batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            return batch_gather(batch, indices)

        else:
            return  None

    def process_batch(self, sess, data):
        """
        Processes batched rollouts.
        Makes every experience independent, enabling further shuffling or sampling.

        Args:
            sess:               tf session obj.
            data (dict):        data dictionary
            is_train (bool):    is data provided are train or test

        Returns:
            on-policy [, off-policy and rp] processed batched data.

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

        return on_policy_batch, off_policy_batch, rp_batch

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


class AMLDG_d(AMLDG):
    """
    Tragectories2distributions mod of AMLDG()
    """

    def __init__(
            self,
            aac_class_ref=SubAAC_d,
            train_support=1,
            num_train_updates=1,
            train_batch_size=64,
            name='AMLDGd',
            **kwargs
    ):
        super(AMLDG_d, self).__init__(aac_class_ref=aac_class_ref, name=name, **kwargs)
        self.train_batch_size = train_batch_size
        self.train_support = train_support
        self.train_distribution_size = int(self.train_support / self.rollout_length)
        self.num_train_updates = num_train_updates
        self.dummy_data = {
            'ep_summary': [None],
            'test_ep_summary': [None],
            'render_summary': [None],
        }

    def process_source(self, sess, train_data_config, test_data_config):
        """
        Trains on single meta-test episode from source domain.

        Args:
            sess:
            train_data_config:
            test_data_config:

        Returns:

        """

        # Collect batch of train rollouts:
        train_batch = self.train_aac.get_batch(
            size=self.train_distribution_size,
            require_terminal=True,
            same_trial=True,
            data_sample_config=train_data_config
        )

        # Process to get train data distribution:
        on_policy_distr, off_policy_distr, rp_distr = self.train_aac.process_batch(sess, train_batch)

        # self.log.warning(
        #     'Train distribution ok, made from ~{} experiences'.format(
        #         len(train_batch['on_policy']) * self.rollout_length
        #     )
        # )

        # Data leakage tests:
        train_trial_chksum = np.average(
            [
                np.average(data['state']['metadata']['trial_num']) for data in train_batch['on_policy']
            ]
        )
        # self.log.warning('train_trial_chksum: {}'.format(train_trial_chksum))
        try:
            assert (np.asarray(
                [
                    (np.asarray(data['state']['metadata']['type']) == 0).all() for data in train_batch['on_policy']
                ]
            ) == 1).all()

        except AssertionError:
            self.log.warning('Train data type mismatch found!')

        # Start collecting meta-test episode data rollout by rollout:
        done = False
        roll_num = 0

        while not done:
            sess.run(self.test_aac.sync_pi_global)  # from global to pi_prime
            sess.run(self.train_aac.sync_pi_local)  # from pi_prime to pi
            feed_dict = {}

            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            for i in range(self.num_train_updates):
                # Sample from train distribution, make feed dictionary:
                feed_dict = self.train_aac._get_main_feeder(
                    sess,
                    self.train_aac.sample_batch(on_policy_distr, self.train_batch_size),
                    self.train_aac.sample_batch(off_policy_distr, self.train_batch_size),
                    self.train_aac.sample_batch(rp_distr, self.train_batch_size),
                    is_train=True,
                    pi=self.train_aac.local_network,
                )

                # Perform pi-prime update conditioned on train sample:
                # note: learn rate is not annealed here
                if wirte_model_summary:
                    fetches = [self.train_op, self.train_aac.model_summary_op]
                else:
                    fetches = [self.train_op]

                fetched = sess.run(fetches, feed_dict=feed_dict)

                # self.log.warning('Pi_prime update {} ok.'.format(i))

            # Note: reusing last train data sample for meta-update step:

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
                    train_trial_chksum,
                    test_trial_chksum
                )
                msg2 = 'Train data config: {}\n Test data config: {}'.format(train_data_config, test_data_config)

                self.log.warning(msg)
                self.log.warning(msg2)

            try:
                assert (np.asarray(test_data['on_policy'][0]['state']['metadata']['type']) == 1).all()

            except AssertionError:
                self.log.warning('Train data type mismatch found!')

            # Perform meta_update using both on_policy test data and sampled train data:

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

            if wirte_model_summary:
                meta_model_summary = meta_fetched[-2]
                model_summary = fetched[-1]

            else:
                meta_model_summary = None
                model_summary = None

            # Summaries etc:
            self.test_aac.process_summary(sess, test_data, meta_model_summary)
            self.train_aac.process_summary(sess, self.dummy_data, model_summary)
            self.local_steps += 1
            roll_num += 1

    def process_target(self, sess, train_data_config, test_data_config):
        """
        Evaluates single meta-test episode from target domain.

        Args:
            sess:
            train_data_config:
            test_data_config:

        Returns:

        """

        done = False
        # Master runner to sample new trial:
        _ = self.train_aac.get_data(data_sample_config=train_data_config, force_new_episode=True)
        while not done:
            # Collect test rollout:
            test_data = self.test_aac.get_data(data_sample_config=test_data_config)
            new_episode = False

            # If meta-test episode has just ended?
            done = np.asarray(test_data['terminal']).any()

            # Summaries:
            # self.log.warning(
            #     'test_data:\n>>{}<<\n>>{}<<\n>>{}<<\nis_test: {}'. format(
            #         test_data['ep_summary'], test_data['test_ep_summary'], test_data['terminal'], test_data['is_test']
            #     )
            # )
            self.test_aac.process_summary(sess, test_data, None)

    def process(self, sess):
        try:
            # Sync parameters:
            sess.run(self.test_aac.sync_pi_global)  # from global to pi_prime
            sess.run(self.train_aac.sync_pi_local)  # from pi_prime to pi

            # Define trial parameters:
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., samples trial
            test_data_config = self.train_aac.get_sample_config(mode=0)  # slave env, catches up with same trial

            # self.log.warning('train_data_config: {}'.format(train_data_config))
            # self.log.warning('test_data_config: {}'.format(test_data_config))

            is_target = train_data_config['trial_config']['sample_type']

            # self.log.warning('is_target: {}'.format(is_target))

            if is_target:
                self.process_target(sess, train_data_config, test_data_config)

            else:
                self.process_source(sess, train_data_config, test_data_config)

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def __process(self, sess):
        try:

            # Sync parameters:
            sess.run(self.train_aac.sync_pi)

            # Define trial parameters:
            train_data_config = self.train_aac.get_sample_config(mode=1)  # master env., samples trial
            test_data_config = self.train_aac.get_sample_config(mode=0)  # slave env, catches up with same trial

            is_target = train_data_config['trial_config']['sample_type']

            # Collect batch of train rollouts:
            train_batch = self.train_aac.get_batch(
                size=self.train_distribution_size,
                require_terminal=True,
                same_trial=True,
                data_sample_config=train_data_config
            )

            # Process to get train data distribution:
            on_policy_distr, off_policy_distr, rp_distr = self.train_aac.process_batch(sess, train_batch)

            # self.log.warning(
            #     'Train distribution ok, made from ~{} experiences'.format(
            #         len(train_batch['on_policy']) * self.rollout_length
            #     )
            # )

            # Data leakage tests:
            train_trial_chksum = np.average(
                [
                    np.average(data['state']['metadata']['trial_num']) for data in train_batch['on_policy']
                ]
            )
            # self.log.warning('train_trial_chksum: {}'.format(train_trial_chksum))
            try:
                assert (np.asarray(
                    [
                        (np.asarray(data['state']['metadata']['type']) == 0).all() for data in train_batch['on_policy']
                    ]
                ) == 1).all()

            except AssertionError:
                self.log.warning('Train data type mismatch found!')

            # Start collecting meta-test episode data rollout by rollout:
            done = False
            roll_num = 0

            while not done:
                # Sync pi_prime from global parameters:
                sess.run(self.test_aac.sync_pi_global)

                wirte_model_summary = \
                    self.local_steps % self.model_summary_freq == 0

                for i in range(self.num_train_updates):
                    # Sample from train distribution, make feed dictionary:
                    feed_dict = self.train_aac._get_main_feeder(
                        sess,
                        self.train_aac.sample_batch(on_policy_distr, self.train_batch_size),
                        self.train_aac.sample_batch(off_policy_distr, self.train_batch_size),
                        self.train_aac.sample_batch(rp_distr, self.train_batch_size),
                        is_train=True,
                        pi=self.train_aac.local_network,
                    )

                    # Perform pi-prime update conditioned on train sample:
                    # note: learn rate is not annealed here
                    if wirte_model_summary:
                        fetches = [self.train_op, self.train_aac.model_summary_op]
                    else:
                        fetches = [self.train_op]

                    fetched = sess.run(fetches, feed_dict=feed_dict)

                    # self.log.warning('Pi_prime update {} ok.'.format(i))

                # Note: reusing last train data sample for meta-update step:

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
                        train_trial_chksum,
                        test_trial_chksum
                        )
                    msg2 = 'Train data config: {}\n Test data config: {}'.format(train_data_config, test_data_config)

                    self.log.warning(msg)
                    self.log.warning(msg2)

                try:
                    assert (np.asarray(test_data['on_policy'][0]['state']['metadata']['type']) == 1).all()

                except AssertionError:
                    self.log.warning('Train data type mismatch found!')

                # Perform meta_update using both on_policy test data and sampled train data:
                if not is_target:
                    # Process test data and perform meta-optimisation step:
                    feed_dict.update(
                        self.test_aac.process_data(sess,test_data,is_train=True, pi=self.test_aac.local_network)
                    )

                    if wirte_model_summary:
                        meta_fetches = [self.meta_train_op, self.test_aac.model_summary_op, self.inc_step]
                    else:
                        meta_fetches = [self.meta_train_op, self.inc_step]

                    meta_fetched = sess.run(meta_fetches, feed_dict=feed_dict)

                    # self.log.warning('Meta-gradients ok.')
                else:
                    # True test, no updates sent to global parameter server:
                    meta_fetched = [None, None]

                    # self.log.warning('Target rollout ok.')

                if wirte_model_summary:
                    meta_model_summary = meta_fetched[-2]
                    model_summary = fetched[-1]

                else:
                    meta_model_summary = None
                    model_summary = None

                # Summaries etc:
                self.test_aac.process_summary(sess, test_data, meta_model_summary)
                self.train_aac.process_summary(sess, self.dummy_data, model_summary)
                self.local_steps += 1
                roll_num += 1
        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)
