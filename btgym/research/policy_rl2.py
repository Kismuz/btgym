import tensorflow as tf
from btgym.algorithms.policy.base import Aac1dPolicy


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
        Basically, RNN context is reset if:
            - episode  initial `state` `trial_num` metadata has been changed form last call (new train trial started);
            - episode metatdata `type` is non-zero (test episode);
            - no context arg is provided (initial episode of training);
        Else, it assumes this is new episode of same train trial has started and carries context on to new episode;

        Episode metadata are provided by DataTrialIterator, which is shaping Trial data distribution in this case,
        and delivered through env.strategy as separate key in observation dictionary.

        Args:
            state:      initial episode state (result of env.reset())
            context:    last previous episode RNN state (last_context of runner)

        Returns:
            RNN zero-state tuple if new trial is detected or same `context`.
        """
        try:
            if state['metadata']['trial_num'] != self.current_trial_num or context is None or state['metadata']['type']:
                # Assume new/initial trial or test, reset context:
                sess = tf.get_default_session()
                new_context = sess.run(self.on_lstm_init_state)
                print('RL^2 policy context reset')

            else:
                # Asssume same training trial, keep context chain:
                new_context = context
            # Keep trial number:
            self.current_trial_num = state['metadata']['trial_num']

        except KeyError:
            raise KeyError(
                'RL^2 policy: expected observation state dict. to have keys [`metadata`]:[`trial_num`,`type`]; got: {}'.
                format(state.keys())
            )

        return new_context
