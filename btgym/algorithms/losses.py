import tensorflow as tf
import  numpy as np
from btgym.algorithms.math_util import cat_entropy, kl_divergence


def aac_loss_def(act_target, adv_target, r_target, pi_logits, pi_vf, entropy_beta, name='aac', verbose=False):
    """
    Advantage Actor Critic loss definition.
    Paper: https://arxiv.org/abs/1602.01783

    Args:
        act_target      tensor holding policy actions targets;
        adv_target      tensor holding policy estimated advantages targets;
        r_target        tensor holding policy empirical returns targets;
        pi__logits      policy logits output tensor;
        pi_vf           policy value function output tensor; 
        entropy_beta    entropy regularization constant;
        name            scope;
        verbose         summary level.

    Returns:
        tensor holding estimated AAC loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name):
        neg_pi_log_prob = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi_logits,
            labels=act_target
        )
        pi_loss = tf.reduce_mean(neg_pi_log_prob * adv_target)
        vf_loss = 0.5 * tf.losses.mean_squared_error(r_target, pi_vf)
        entropy = tf.reduce_mean(cat_entropy(pi_logits))

        loss = pi_loss + vf_loss - entropy * entropy_beta

        summaries = [
            tf.summary.scalar('policy_loss', pi_loss),
            tf.summary.scalar('value_loss', vf_loss),
        ]
        if verbose:
            summaries += [tf.summary.scalar('entropy', entropy)]

    return loss, summaries


def ppo_loss_def(act_target, adv_target, r_target, pi_logits, pi_vf, pi_old_logits, entropy_beta, epsilon,
                 name='ppo', verbose=False):
    """
    PPO clipped surrogate loss definition, as (7) in https://arxiv.org/pdf/1707.06347.pdf

    Args:
        act_target      tensor holding policy actions targets;
        adv_target      tensor holding policy estimated advantages targets;
        r_target        tensor holding policy empirical returns targets;
        pi__logits      policy logits output tensor;
        pi_vf           policy value function output tensor;       
        pi_old_logits   old_policy logits output tensor;
        entropy_beta    entropy regularization constant
        epsilon         L^Clip epsilon tensor;
        name            scope;
        verbose         summary level.

    Returns:
        tensor holding estimated PPO L^Clip loss;
        list of related tensorboard summaries.
    """
    #act_target = tf.placeholder(tf.float32, [None, env.action_space.n], name="on_policy_action_pl")
    #adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
    #r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")
    with tf.name_scope(name):
        pi_log_prob = - tf.nn.softmax_cross_entropy_with_logits(
            logits=pi_logits,
            labels=act_target
        )
        pi_old_log_prob = tf.stop_gradient(
            - tf.nn.softmax_cross_entropy_with_logits(
                logits=pi_old_logits,
                labels=act_target
            )
        )
        pi_ratio = tf.exp(pi_log_prob - pi_old_log_prob)

        surr1 = pi_ratio * adv_target  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(pi_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv_target

        pi_surr_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.losses.mean_squared_error(r_target, pi_vf)  # V.fn. loss
        entropy = tf.reduce_mean(cat_entropy(pi_logits))

        loss = pi_surr_loss + vf_loss - entropy * entropy_beta

        # Info:
        mean_pi_ratio = tf.reduce_mean(pi_ratio)
        mean_vf = tf.reduce_mean(pi_vf)
        mean_kl_old_new = tf.reduce_mean(kl_divergence(pi_old_logits, pi_logits ))

        summaries = [
            tf.summary.scalar('l_clip_loss', pi_surr_loss),
            tf.summary.scalar('value_loss', vf_loss),
        ]
        if verbose:
            summaries += [
                tf.summary.scalar('entropy', entropy),
                tf.summary.scalar('Dkl_old_new', mean_kl_old_new),
                tf.summary.scalar('pi_ratio', mean_pi_ratio),
                tf.summary.scalar('value_f', mean_vf),
            ]

    return loss, summaries


def value_fn_loss_def(r_target, pi_vf, name='value_replay', verbose=False):
    """
    Value function loss.

    Args:
        r_target        tensor holding policy empirical returns targets;
        pi_vf           policy value function output tensor;
        name            scope;
        verbose         summary level.

    Returns:
        tensor holding estimated value fn. loss;
        list of related tensorboard summaries.
    """
    # r_target = tf.placeholder(tf.float32, [None], name="vr_target")
    with tf.name_scope(name):
        loss = tf.losses.mean_squared_error(r_target, pi_vf)

        if verbose:
            summaries = [tf.summary.scalar('v_loss', loss)]
        else:
            summaries = []

    return loss, summaries


def pc_loss_def(actions, targets, pi_pc_q, name='pixel_control', verbose=False):
    """
    Pixel control auxiliary task loss definition.
    Paper: https://arxiv.org/abs/1611.05397
    Borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
    https://miyosuda.github.io/
    https://github.com/miyosuda/unreal

    Args:
        actions     tensor holding policy actions;
        targets     tensor holding estimated pixel-change targets;
        pi_pc_q     policy Q-value features output tensor;
        name        scope;
        verbose     summary level.

    Returns:
        tensor holding estimated pc loss;
        list of related tensorboard summaries.
    """
    #actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="pc_action")
    #targets = tf.placeholder(tf.float32, [None, None, None], name="pc_target")
    with tf.name_scope(name):
        # Get Q-value features for actions been taken and define loss:
        pc_action_reshaped = tf.reshape(actions, [-1, 1, 1, tf.shape(actions)[-1]])
        pc_q_action = tf.multiply(pi_pc_q, pc_action_reshaped)
        pc_q_action = tf.reduce_sum(pc_q_action, axis=-1, keep_dims=False)

        batch_size = tf.shape(targets)[0]
        loss = tf.reduce_sum(tf.square(targets - pc_q_action)) / tf.cast(batch_size, tf.float32)
        #loss = tf.losses.absolute_difference(targets, pc_q_action)
        if verbose:
            summaries = [tf.summary.scalar('q_loss', loss)]
        else:
            summaries = []

    return loss, summaries


def rp_loss_def(rp_targets, pi_rp_logits, name='reward_prediction', verbose=False):
    """
    Reward prediction auxillary task loss definition.
    Paper: https://arxiv.org/abs/1611.05397
    Borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
    https://miyosuda.github.io/
    https://github.com/miyosuda/unreal

    Args:
        targets         tensor holding reward prediction target;
        pi_rp_logits    policy reward predictions tensor;
        name             scope;
        verbose          summary level.

    Returns:
        tensor holding estimated rp loss;
        list of related tensorboard summaries.
    """
    #rp_targets = tf.placeholder(tf.float32, [1, 3], name="rp_target")
    with tf.name_scope(name):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=rp_targets,
            logits=pi_rp_logits
        )[0]
        if verbose:
            summaries = [tf.summary.scalar('class_loss', loss), ]
        else:
            summaries = []

    return loss, summaries


def aac_ppo_loss_def(act_target, adv_target, r_target, pi_logits, pi_vf, pi_old_logits, entropy_beta, epsilon,
                 name='ppo', verbose=False):
    """

    Note:
        IGNORE!!

    Args:
        act_target      tensor holding policy actions targets;
        adv_target      tensor holding policy estimated advantages targets;
        r_target        tensor holding policy empirical returns targets;
        pi__logits      policy logits output tensor;
        pi_vf           policy value function output tensor;
        pi_old_logits   old_policy logits output tensor;
        entropy_beta    entropy regularization constant
        epsilon         L^Clip epsilon tensor;
        name            scope;
        verbose         summary level.
    Returns:
        tensor holding estimated PPO L^Clip loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name):
        pi_log_prob = - tf.nn.softmax_cross_entropy_with_logits(
            logits=pi_logits,
            labels=act_target
        )
        pi_old_log_prob = tf.stop_gradient(
            - tf.nn.softmax_cross_entropy_with_logits(
                logits=pi_old_logits,
                labels=act_target
            )
        )
        pi_ratio = tf.exp(pi_log_prob - pi_old_log_prob)

        surr1 = pi_ratio * adv_target  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(pi_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv_target

        pi_surr_loss = - 0.5 * tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = 0.75 * tf.reduce_mean(tf.square(pi_vf - r_target))  # V.fn. loss
        entropy = tf.reduce_mean(cat_entropy(pi_logits))

        aac_loss = - 0.5 * tf.reduce_mean(pi_log_prob * adv_target)

        loss = pi_surr_loss + aac_loss + vf_loss - entropy * entropy_beta

        aac_surr_loss = aac_loss + pi_surr_loss

        # Info:
        mean_pi_ratio = tf.reduce_mean(pi_ratio)
        mean_vf = tf.reduce_mean(pi_vf)
        mean_kl_old_new = tf.reduce_mean(kl_divergence(pi_old_logits, pi_logits ))

        summaries = [
            tf.summary.scalar('l_clip_loss', pi_surr_loss),
            tf.summary.scalar('value_loss', vf_loss),
            tf.summary.scalar('pi_loss', aac_loss),
            tf.summary.scalar('aac_ppo_loss', aac_surr_loss),
        ]
        if verbose:
            summaries += [
                tf.summary.scalar('entropy', entropy),
                tf.summary.scalar('Dkl_old_new', mean_kl_old_new),
                tf.summary.scalar('pi_ratio', mean_pi_ratio),
                tf.summary.scalar('value_f', mean_vf),
            ]

    return loss, summaries