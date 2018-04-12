import tensorflow as tf
import time

import sys
from logbook import Logger, StreamHandler

from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class MetaAAC_2_0(GuidedAAC):
    """
    RNN adaptation experiment
    """

    def __init__(
            self,
            trial_source_target_cycle=(1, 0),
            num_episodes_per_trial=1,
            name='AAC_FWRNN_Ada',
            **kwargs
    ):
        runner_config = {
            'class_ref': BaseSynchroRunner,
            'kwargs': {
                'data_sample_config': {'mode': 0},
                'name': '',
            },
        }
        super(MetaAAC_2_0, self).__init__(
            runner_config=runner_config,
            name=name,
            **kwargs
        )

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

    def get_sample_config(self, mode=0, **kwargs):
        """
        Returns environment configuration parameters for next episode to sample.

        Args:
              mode:     bool, False for slave (train data), True for master (test data)

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """

        new_trial = 0

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

    def get_episode(self, **kwargs):
        data_streams = [runner.get_episode(**kwargs) for runner in self.runners]
        return {key: [stream[key] for stream in data_streams] for key in data_streams[0].keys()}

    def process(self, sess, **kwargs):
        if self.task < 1:
            self.process_test(sess)

        else:
            self.process_train(sess)

    def process_test(self, sess):
        """
         test step.

         Args:
             sess (tensorflow.Session):   tf session obj.

         """
        # Copy from parameter server:
        sess.run(self.sync_pi)
        for i in range(1):
            test_data = self.get_episode(init_context=0)
            self.process_summary(sess, test_data)
            #self.log.warning('self.current_episode: {}'.format(self.current_episode))

        #time.sleep(5)

    def process_train(self, sess):
        """
        Train step.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Say `No` to redundant summaries:
            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            # Collect train trajectory:
            train_data = self.get_data()
            feed_dict = self.process_data(sess,,,,, train_data,,
                        # self.log.warning('Train data ok.')

                        # Copy from parameter server:
                        sess.run(self.sync_pi)
            # self.log.warning('Sync ok.')

            # Update pi_prime parameters wrt collected data:
            if wirte_model_summary:
                fetches = [self.train_op, self.model_summary_op, self.inc_step]
            else:
                fetches = [self.train_op, self.inc_step]

            fetched = sess.run(fetches, feed_dict=feed_dict)

            # self.log.warning('Train gradients ok.')

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            # Write down summaries:
            self.process_summary(sess, train_data, model_summary)
            self.local_steps += 1

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)



