import tensorflow as tf
import numpy as np
from btgym.research.mldg.aac_1 import AMLDG_1


class AMLDG_1a(AMLDG_1):
    """
    AMLDG_1 + learnable inner adaptation step
    """

    def __init__(self, name='AMLDG1a', **kwargs):

        super(AMLDG_1, self).__init__(name=name, **kwargs)

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

        raw_pi_grads = tf.gradients(self.meta_train_loss, pi.var_list)

        # Clipped gradients:
        pi.grads, _ = tf.clip_by_global_norm(
            raw_pi_grads,
            40.0
        )
        pi_prime.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.meta_test_loss, pi_prime.var_list),
            40.0
        )

        # Adapting fast opt. rate:


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
        meta_model_summaries = [
            tf.summary.scalar('meta_grad_global_norm', tf.global_norm(self.grads)),
            tf.summary.scalar('total_meta_loss', self.loss),
            #tf.summary.scalar('alpha_learn_rate', self.alpha_rate),
            #tf.summary.scalar('alpha_learn_rate_loss', self.alpha_rate_loss)
        ]
        return meta_model_summaries
