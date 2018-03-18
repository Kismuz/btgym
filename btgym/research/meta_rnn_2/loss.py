import tensorflow as tf
import  numpy as np
from btgym.algorithms.math_utils import cat_entropy

def meta_loss_def_1_0(
        act_target_train,
        act_target_test,
        adv_target_train,
        adv_target_test,
        r_target_train,
        r_target_test,
        pi_logits_train,
        pi_logits_test,
        pi_vf_train,
        pi_vf_test,
        pi_prime_logits,
        entropy_beta,
        epsilon=None,
        name='_meta_',
        verbose=False
):
    with tf.name_scope(name + '/meta'):
        neg_pi_log_prob_train = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_logits_train,
            labels=act_target_train
        )
        neg_pi_log_prob_test = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_logits_test,
            labels=act_target_test
        )
        pi_loss = tf.reduce_mean(
            (neg_pi_log_prob_train + neg_pi_log_prob_test) * adv_target_test
        )
        vf_loss_train = 0.5 * tf.losses.mean_squared_error(r_target_test, pi_vf_train)
        vf_loss_test = 0.5 * tf.losses.mean_squared_error(r_target_test, pi_vf_test)

        entropy = tf.reduce_mean(cat_entropy(pi_logits_test))

        loss = pi_loss + vf_loss_test + vf_loss_train - entropy * entropy_beta

        mean_vf_test = tf.reduce_mean(pi_vf_test)
        mean_vf_train = tf.reduce_mean(pi_vf_train)

        summaries = [
            tf.summary.scalar('meta_policy_loss', pi_loss),
            tf.summary.scalar('meta_value_loss_test', vf_loss_test),
        ]
        if verbose:
            summaries += [
                tf.summary.scalar('entropy', entropy),
                tf.summary.scalar('value_fn_test', mean_vf_test),
                tf.summary.scalar('value_fn_train', mean_vf_train)
            ]

    return loss, summaries
