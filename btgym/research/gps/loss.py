import tensorflow as tf


def guided_aac_loss_def_0_0(pi_actions, expert_actions, name='on_policy/aac', verbose=False, **kwargs):
    """
    Cross-entropy imitation loss on expert actions.

    Args:
        pi_actions:             tensor holding policy actions logits
        expert_actions:         tensor holding expert actions probability distribution
        name:           loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        # Loss over expert action's distribution:

        neg_pi_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_actions,
            labels=tf.argmax(expert_actions, axis=-1)
        )
        loss = tf.reduce_mean(neg_pi_log_prob)

        if verbose:
            summaries = [tf.summary.scalar('actions_ce', loss)]
        else:
            summaries = []

    return loss, summaries


def guided_aac_loss_def_0_1(pi_actions, expert_actions, name='on_policy/aac', verbose=False, **kwargs):
    """
    Cross-entropy imitation loss on {`buy`, `sell`} subset of expert actions.

    Args:
        pi_actions:             tensor holding policy actions logits
        expert_actions:         tensor holding expert actions probability distribution
        name:           loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        # Loss over expert buy/ sell:
        # Cross-entropy on subset?...

        neg_pi_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pi_actions[..., 1:3],
            labels=expert_actions[..., 1:3]
        )
        loss = tf.reduce_mean(neg_pi_log_prob)

        if verbose:
            summaries = [tf.summary.scalar('actions_ce', loss)]
        else:
            summaries = []

    return loss, summaries


def guided_aac_loss_def_0_3(pi_actions, expert_actions, name='on_policy/aac', verbose=False, **kwargs):
    """
    MSE imitation loss on {`buy`, `sell`} subset of expert actions.

    Args:
        pi_actions:             tensor holding policy actions logits
        expert_actions:         tensor holding expert actions probability distribution
        name:           loss op name scope

    Returns:
        tensor holding estimated imitation loss;
        list of related tensorboard summaries.
    """
    with tf.name_scope(name + '/guided_loss'):
        if 'guided_lambda' in kwargs.keys():
            guided_lambda = kwargs['guided_lambda']
        else:
            guided_lambda = 1.0

        # Loss over expert buy/ sell:
        loss = tf.losses.mean_squared_error(
            labels=expert_actions[..., 1:3],
            predictions=tf.nn.softmax(pi_actions)[..., 1:3],
        ) * guided_lambda

        if verbose:
            summaries = [tf.summary.scalar('actions_mse', loss)]
        else:
            summaries = []

    return loss, summaries
