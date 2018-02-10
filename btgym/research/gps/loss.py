import tensorflow as tf
#import  numpy as np
from btgym.algorithms.math_utils import cat_entropy, kl_divergence


def guided_aac_loss_def_0_0(pi, mu,  entropy_beta, name='on_policy/aac', **kwargs):
    """
    Imitation loss on expert actions.
    Args:
        pi:     trainable policy
        mu:     expert policy
        entropy_beta:   entropy reg. constant
        name:   loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        # ~ batch-act deterministically:
        expert_actions = tf.argmax(mu.on_logits, axis=-1)
        neg_pi_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi.on_logits,
            labels=expert_actions
        )
        pi_loss = tf.reduce_mean(neg_pi_log_prob)
        entropy = tf.reduce_mean(cat_entropy(pi.on_logits))
        loss = pi_loss - entropy_beta * entropy
        summaries = [
            tf.summary.scalar('policy', pi_loss),
        ]
    return loss, summaries


def guided_aac_loss_def_0_1(pi, mu, entropy_beta, name='on_policy/aac'):
    """
    Imitation loss on expert actions (exclusive) + value fn.
    Args:
        pi:             trainable policy
        mu:             expert policy
        entropy_beta:   entropy reg. constant
        name:           loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        # ~ batch-act deterministically:
        expert_actions = tf.argmax(mu.on_logits, axis=-1)
        neg_pi_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi.on_logits,
            labels=expert_actions
        )
        pi_loss = tf.reduce_mean(neg_pi_log_prob)
        v_loss = 0.5 * tf.losses.mean_squared_error(mu.on_vf, pi.on_vf)
        entropy = tf.reduce_mean(cat_entropy(pi.on_logits))
        loss = pi_loss + v_loss - entropy_beta * entropy
        summaries = [
            tf.summary.scalar('policy', pi_loss),
            tf.summary.scalar('value', v_loss)
        ]
    return loss, summaries


def guided_aac_loss_def_0_2(pi, mu, entropy_beta, name='on_policy/aac'):
    """
    Imitation loss on expert actions, distribution + value fn.
    Args:
        pi:             trainable policy
        mu:             expert policy
        entropy_beta:   entropy reg. constant
        name:           loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        # Loss over expert action's distribution:
        expert_actions_distr = tf.nn.softmax(mu.on_logits)
        neg_pi_log_prob = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi.on_logits,
            labels=expert_actions_distr
        )
        pi_loss = tf.reduce_mean(neg_pi_log_prob)
        v_loss = 0.5 * tf.losses.mean_squared_error(mu.on_vf, pi.on_vf)
        entropy = tf.reduce_mean(cat_entropy(pi.on_logits))
        loss = pi_loss + v_loss - entropy_beta * entropy
        summaries = [
            tf.summary.scalar('policy', pi_loss),
            tf.summary.scalar('value', v_loss)
        ]
    return loss, summaries