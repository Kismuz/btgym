import tensorflow as tf
from btgym.research.gps.policy import GuidedPolicy_0_0
import numpy as np


class AacStackedMetaPolicy(GuidedPolicy_0_0):
    """
    Enabling meta-learning by embedding learning algorithm in RNN activations.

    Use in conjunction with btgym.research.meta_rnn_2.env_runner.MetaEnvRunnerFn;

    Paper:
    `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING`,
    https://arxiv.org/pdf/1611.02779.pdf

    `Get_initial_features()` method has been modified to enamble meta-learning loop:
    either to reset RNN critic (lstm_2) context to zero-state or return continue from the end of previous episode,
    depending on episode metadata received.
    """
    def __init__(self, **kwargs):
        super(AacStackedMetaPolicy, self).__init__(**kwargs)
        self.current_trial_num = -1  # always give initial context at first call

    def get_initial_features(self, state, context=None):
        """
        Returns initial RNN context
        RNN_1 (lower, actor) context is reset at every call.
        RNN_2 (upper, critic) context is reset if :
            - episode  initial `state` `trial_num` metadata has been changed form last call (new trial started);
            - no context arg is provided (initial episode of training);
        ... else carries critic context on to new episode;

        Episode metadata are provided by DataTrialIterator, which is shaping Trial data distribution in this case,
        and delivered through env.strategy as separate key in observation dictionary.

        Args:
            state:      initial episode state (result of env.reset())
            context:    last previous episode RNN state (last_context of runner)

        Returns:
            2 level RNN zero-state tuple.

        Raises:
            KeyError if [`metadata`]:[`trial_num`,`type`] keys not found
        """
        #print('Meta_policy_init_metadata:', state['metadata'])
        #print('self.current_trial_num:', self.current_trial_num)
        try:
            sess = tf.get_default_session()
            new_context = list(sess.run(self.on_lstm_init_state))
            if context is not None:
                if state['metadata']['trial_num'] == self.current_trial_num or state['metadata']['type']:
                    # Asssume same training trial or test episode pass, critic context intact to new episode:
                    new_context[-1] = context[-1]
                    # TODO: !
                    # FULL context intact to new episode:
                    #new_context = context

                    #print('Meta_policy Actor context reset')
                else:
                    #print('Meta_policy Actor and Critic context reset')
                    pass
            # Back to tuple:
            new_context = tuple(new_context)
            # Keep trial number:
            self.current_trial_num = state['metadata']['trial_num']

        except KeyError:
            raise KeyError(
                'Meta_policy: expected observation state dict. to have keys [`metadata`]:[`trial_num`,`type`]; got: {}'.
                format(state.keys())
            )

        return new_context

