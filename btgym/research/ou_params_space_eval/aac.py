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
            name='OUpA3C',
            **kwargs
    ):
        super(OUpAAC, self).__init__(
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