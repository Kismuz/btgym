import tensorflow as tf
from btgym.algorithms.policy import Aac1dPolicy


class AacRL2Policy(Aac1dPolicy):
    """
    This policy class in conjunction with DataTrialIterator classes from btgym.datafeed
    is aimed to implement RL^2 algorithm by Duan et al.

    Paper:
    `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING`,
    https://arxiv.org/pdf/1611.02779.pdf

    The only difference from Base policy is `get_initial_features()` method, which has been changed
    either to reset RNN context to zero-state or return context from the end of previous episode,
    depending on episode metadata received.
    """
    def __init__(self, **kwargs):
        super(AacRL2Policy, self).__init__(**kwargs)
        self.current_trial_num = -1  # always give initial context at first call

    def get_initial_features(self, state, context=None):
        """
        Returns RNN initial context.
        Basically, it looks for episode `trial_num` metadata in initial `state` and
        if `trial_num` has been changed form last method call - RNN context is reset;
        else it assumes this is new episode of same trial and carry context on to new episode;
        if no context arg is provided - assumes it is initial episode of training and resets.

        Episode metadata are provided by DataTrialIterator, which is shaping Trial data distribution in this case,
        and delivered through env.strategy as separate key in observation dictionary.

        Args:
            state:      initial episode state (result of env.reset())
            context:    last previous episode RNN state (last_context of runner)

        Returns:
            RNN zero-state tuple if new trial is detected or same `context`.
        """
        try:
            if state['metadata']['trial_num'] != self.current_trial_num or context is None:
                # Assume new or initial trial, reset context:
                sess = tf.get_default_session()
                new_context = sess.run(self.on_lstm_init_state)
                print('RL^2 policy context reset')

            else:
                # Asssume same trial, keep context chain:
                new_context = context

            self.current_trial_num = state['metadata']['trial_num']

        except KeyError:
            raise KeyError('RL^2 policy: expected observation state dict. to have keys `metadata`-`trial_num`, got: {}'.
                           format(state.keys()))

        return new_context
