import tensorflow as tf
import  numpy as np
from btgym.algorithms.math_utils import cat_entropy, kl_divergence


def aac_loss_def(act_target, adv_target, r_target, pi_logits, pi_vf, pi_prime_logits,
                 entropy_beta, epsilon=None, name='_aac_', verbose=False):
    """
    Advantage Actor Critic loss definition.
    Paper: https://arxiv.org/abs/1602.01783

    Args:
        act_target:      tensor holding policy actions targets;
        adv_target:      tensor holding policy estimated advantages targets;
        r_target:        tensor holding policy empirical returns targets;
        pi_logits:       policy logits output tensor;
        pi_prime_logits: not used;
        pi_vf:           policy value function output tensor;
        entropy_beta:    entropy regularization constant;
        epsilon:         not used;
        name:            scope;
        verbose:         summary level.

    Returns:
        tensor holding estimated AAC loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/aac'):
        neg_pi_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_logits,
            labels=act_target
        )
        pi_loss = tf.reduce_mean(neg_pi_log_prob * adv_target)
        vf_loss = 0.5 * tf.losses.mean_squared_error(r_target, pi_vf)
        entropy = tf.reduce_mean(cat_entropy(pi_logits))

        loss = pi_loss + vf_loss - entropy * entropy_beta

        mean_vf = tf.reduce_mean(pi_vf)
        mean_t_target = tf.reduce_mean(r_target)

        summaries = [
            tf.summary.scalar('policy_loss', pi_loss),
            tf.summary.scalar('value_loss', vf_loss),
        ]
        if verbose:
            summaries += [
                tf.summary.scalar('entropy', entropy),
                tf.summary.scalar('value_fn', mean_vf),
                # tf.summary.scalar('empirical_return',mean_t_target),
                # tf.summary.histogram('value_fn', pi_vf),
                # tf.summary.histogram('empirical_return', r_target),
            ]

    return loss, summaries


def ppo_loss_def(act_target, adv_target, r_target, pi_logits, pi_vf, pi_prime_logits, entropy_beta, epsilon,
                 name='_ppo_', verbose=False):
    """
    PPO clipped surrogate loss definition, as (7) in https://arxiv.org/pdf/1707.06347.pdf

    Args:
        act_target:      tensor holding policy actions targets;
        adv_target:      tensor holding policy estimated advantages targets;
        r_target:        tensor holding policy empirical returns targets;
        pi_logits:       policy logits output tensor;
        pi_vf:           policy value function output tensor;
        pi_prime_logits: old_policy logits output tensor;
        entropy_beta:    entropy regularization constant
        epsilon:         L^Clip epsilon tensor;
        name:            scope;
        verbose:         summary level.

    Returns:
        tensor holding estimated PPO L^Clip loss;
        list of related tensorboard summaries.
    """
    #act_target = tf.placeholder(tf.float32, [None, env.action_space.n], name="on_policy_action_pl")
    #adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
    #r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")
    with tf.name_scope(name + '/ppo'):
        pi_log_prob = - tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_logits,
            labels=act_target
        )
        pi_old_log_prob = tf.stop_gradient(
            - tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=pi_prime_logits,
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
        mean_kl_old_new = tf.reduce_mean(kl_divergence(pi_prime_logits, pi_logits))

        summaries = [
            tf.summary.scalar('l_clip_loss', pi_surr_loss),
            tf.summary.scalar('value_loss', vf_loss),
        ]
        if verbose:
            summaries += [
                tf.summary.scalar('entropy', entropy),
                tf.summary.scalar('Dkl_old_new', mean_kl_old_new),
                tf.summary.scalar('pi_ratio', mean_pi_ratio),
                tf.summary.scalar('value_fn', mean_vf),
            ]

    return loss, summaries


def value_fn_loss_def(r_target, pi_vf, name='_vr_', verbose=False):
    """
    Value function loss.

    Args:
        r_target:        tensor holding policy empirical returns targets;
        pi_vf:           policy value function output tensor;
        name:            scope;
        verbose:         summary level.

    Returns:
        tensor holding estimated value fn. loss;
        list of related tensorboard summaries.
    """
    # r_target = tf.placeholder(tf.float32, [None], name="vr_target")
    with tf.name_scope(name + '/value_replay'):
        loss = tf.losses.mean_squared_error(r_target, pi_vf)

        if verbose:
            summaries = [tf.summary.scalar('v_loss', loss)]
        else:
            summaries = []

    return loss, summaries


def pc_loss_def(actions, targets, pi_pc_q, name='_pc_', verbose=False):
    """
    Pixel control auxiliary task loss definition.

    Paper: https://arxiv.org/abs/1611.05397

    Borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:

    https://miyosuda.github.io/

    https://github.com/miyosuda/unreal

    Args:
        actions:     tensor holding policy actions;
        targets:     tensor holding estimated pixel-change targets;
        pi_pc_q:     policy Q-value features output tensor;
        name:        scope;
        verbose:     summary level.

    Returns:
        tensor holding estimated pc loss;
        list of related tensorboard summaries.
    """
    #actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="pc_action")
    #targets = tf.placeholder(tf.float32, [None, None, None], name="pc_target")
    with tf.name_scope(name + '/pixel_control'):
        # Get Q-value features for actions been taken and define loss:
        pc_action_reshaped = tf.reshape(actions, [-1, 1, 1, tf.shape(actions)[-1]])
        pc_q_action = tf.multiply(pi_pc_q, pc_action_reshaped)
        pc_q_action = tf.reduce_sum(pc_q_action, axis=-1, keepdims=False)

        batch_size = tf.shape(targets)[0]
        loss = tf.reduce_sum(tf.square(targets - pc_q_action)) / tf.cast(batch_size, tf.float32)
        #loss = tf.losses.absolute_difference(targets, pc_q_action)
        if verbose:
            summaries = [tf.summary.scalar('q_loss', loss)]
        else:
            summaries = []

    return loss, summaries


def rp_loss_def(rp_targets, pi_rp_logits, name='_rp_', verbose=False):
    """
    Reward prediction auxillary task loss definition.

    Paper: https://arxiv.org/abs/1611.05397

    Borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:

    https://miyosuda.github.io/

    https://github.com/miyosuda/unreal


    Args:
        targets:         tensor holding reward prediction target;
        pi_rp_logits:    policy reward predictions tensor;
        name:             scope;
        verbose:          summary level.

    Returns:
        tensor holding estimated rp loss;
        list of related tensorboard summaries.
    """
    #rp_targets = tf.placeholder(tf.float32, [1, 3], name="rp_target")
    with tf.name_scope(name + '/reward_prediction'):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=rp_targets,
            logits=pi_rp_logits
        )[0]
        if verbose:
            summaries = [tf.summary.scalar('class_loss', loss), ]
        else:
            summaries = []

    return loss, summaries


def ae_loss_def(targets, logits, alpha=1.0, name='ae_loss', verbose=False, **kwargs):
    """
    Mean quadratic autoencoder reconstruction loss definition

    Args:
        targets:        tensor holding reconstruction target
        logits:         t ensor holding decoded aa decoder output
        alpha:          loss weight constant
        name:           scope
        verbose:        summary level.

    Returns:
        tensor holding estimated reconstruction loss
        list of summarues
    """
    with tf.name_scope(name + '/ae'):
        loss = tf.losses.mean_squared_error(targets, logits)

        if verbose:
            summaries = [tf.summary.scalar('reconstruct_loss', loss)]
        else:
            summaries = []

        return alpha * loss, summaries


def beta_vae_loss_def(targets, logits, d_kl, alpha=1.0, beta=1.0, name='beta_vae_loss', verbose=False):
    """
    Beta-variational autoencoder loss definition

    Papers:
        http://www.matthey.me/pdf/betavae_iclr_2017.pdf
        https://drive.google.com/file/d/0Bwy4Nlx78QCCNktVTFFMTUs4N2oxY295VU9qV25MWTBQS2Uw/view

    Args:
        targets:
        logits:
        d_kl:
        alpha:
        beta:
        name:
        verbose:

    Returns:
        tensor holding estimated loss
        list of summarues

    """
    with tf.name_scope(name + '/b_vae'):
        r_loss = tf.losses.mean_squared_error(targets, logits)
        vae_loss = tf.reduce_mean(d_kl)
        loss = alpha * r_loss + beta * vae_loss
        if verbose:
            summaries = [
                tf.summary.scalar('reconstruct_loss', r_loss),
                tf.summary.scalar('d_kl_loss', vae_loss),
            ]
        else:
            summaries = []

        return loss, summaries



