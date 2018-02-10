import tensorflow as tf


def guided_aac_loss_def_0_0(pi_actions, expert_actions, name='on_policy/aac', verbose=False):
    """
    Imitation loss on expert actions, distribution + value fn.

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
            labels=expert_actions
        )
        loss = tf.reduce_mean(neg_pi_log_prob)

        if verbose:
            summaries = [tf.summary.scalar('actions_ce', loss)]
        else:
            summaries = []

    return loss, summaries
