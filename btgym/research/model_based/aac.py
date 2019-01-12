import numpy as np
import tensorflow as tf
from btgym.research.gps.aac import GuidedAAC
from .runner import OUpRunner


class OUpAAC(GuidedAAC):
    """
    Extends parent `GuidedAAC` class with additional summaries related to Orn-Uhl. data generating process.
    """

    def __init__(
            self,
            runner_config=None,
            aac_lambda=1.0,
            guided_lambda=0.0,
            name='OUpA3C',
            **kwargs
    ):
        super(OUpAAC, self).__init__(
            aac_lambda=aac_lambda,
            guided_lambda=guided_lambda,
            name=name,
            runner_config={
                    'class_ref': OUpRunner,
                    'kwargs': {}
                },
            **kwargs
        )

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
            ou_lambda=tf.placeholder(tf.float32, ),
            ou_sigma=tf.placeholder(tf.float32, ),
            ou_mu=tf.placeholder(tf.float32, ),

        )
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
                tf.summary.scalar('episode_train/final_value', ep_summary['final_value']),
                tf.summary.scalar('episode_train/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_train/ou_lambda', ep_summary['ou_lambda']),
                tf.summary.scalar('episode_train/ou_sigma', ep_summary['ou_sigma']),
                tf.summary.scalar('episode_train/ou_mu', ep_summary['ou_mu']),
            ],
            name='episode_train_btgym'
        )
        # Test episode stat. summary:
        ep_summary['test_btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_test/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_test/final_value', ep_summary['final_value']),
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


class MA4C(OUpAAC):
    """
    Continuous data model adaptation.
    Single test episode run (yet).
    Master runner is always a tester, others run train episodes only.
    """
    def __init__(self, name='MA4C', **kwargs):
        super(MA4C, self).__init__(name=name, **kwargs)

    def get_sample_config(self, **kwargs):
        """
        Returns environment configuration parameters for next episode to sample.
        Always samples single test episode for chief worker, train episode otherwise.

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """
        if self.task == 0:
            if self.current_test_episode < 1:
                self.current_test_episode += 1
                episode_type = 1

            else:
                msg = 'Currently only single test episode per run is supported.'
                self.log.error(msg)
                raise RuntimeError(msg)

        else:
            episode_type = 0

        self.log.debug('sample type: {}'.format(episode_type))

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=True,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config

    def process(self, sess, **kwargs):
        """
        Main train step method wrapper. Override if needed.

        Args:
            sess (tensorflow.Session):   tf session obj.
            kwargs:                      any


        """
        if self.task == 0:
            self.process_test(sess, **kwargs)

        else:
            self.process_train(sess, **kwargs)

    def process_test(self, sess, **kwargs):
        """
        Processes test rollout.

        Args:
            sess:
            **kwargs:
        """
        try:
            # Collect data from child thread runners,
            # providing sync_op kwarg forces policy update once per step (instead of once per rollout):
            data = self.get_data(policy_sync_op=self.sync_pi)
            self.process_summary(sess, data)

        except:
            msg = 'process_test() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.error(msg)
            raise RuntimeError(msg)

    def process_train(self, sess, **kwargs):
        try:
            # Collect data from child thread runners:
            data = self.get_data()

            # Copy weights from local policy to local target policy:
            if self.use_target_policy and self.local_steps % self.pi_prime_update_period == 0:
                sess.run(self.sync_pi_prime)

            # Extra check: accept no test rollouts here:
            is_train = not np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any()

            if not is_train:
                self.log.error(
                    'got rollout episode. type: {}, trial_type: {}, is_train: {}'.format(
                        np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any(),
                        np.asarray([env['state']['metadata']['trial_type'] for env in data['on_policy']]).any(),
                        is_train
                    )
                )
                msg = 'process_train() unexpectedly got test rollout' + \
                      '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
                self.log.error(msg)
                raise AssertionError(msg)

            sess.run(self.sync_pi)

            feed_dict = self.process_data(sess, data, is_train, self.local_network, self.local_network_prime)

            # Say `No` to redundant summaries:
            wirte_model_summary =\
                self.local_steps % self.model_summary_freq == 0

            #fetches = [self.train_op, self.local_network.debug]  # include policy debug shapes
            fetches = [self.train_op]

            if wirte_model_summary:
                fetches_last = fetches + [self.model_summary_op, self.inc_step]
            else:
                fetches_last = fetches + [self.inc_step]

            # Do a number of SGD train epochs:
            # When doing more than one epoch, we actually use only last summary:
            for i in range(self.num_epochs - 1):
                fetched = sess.run(fetches, feed_dict=feed_dict)

            fetched = sess.run(fetches_last, feed_dict=feed_dict)

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            self.local_steps += 1  # only update on train steps

            # Write down summaries:
            self.process_summary(sess, data, model_summary)

        except:
            msg = 'process_train() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.error(msg)
            raise RuntimeError(msg)