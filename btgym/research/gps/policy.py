import tensorflow as tf
from btgym.algorithms.policy.stacked_lstm import AacStackedRL2Policy
from btgym.algorithms.nn.layers import noisy_linear


class GuidedPolicy_0_0(AacStackedRL2Policy):
    """
    Guided policy: simple configuration wrapper around Stacked LSTM architecture.
    """

    def __init__(
        self,
        conv_2d_layer_config=(
                (32, (3, 1), (2, 1)),
                (32, (3, 1), (2, 1)),
                (64, (3, 1), (2, 1)),
                (64, (3, 1), (2, 1))
            ),
            lstm_class_ref=tf.contrib.rnn.LayerNormBasicLSTMCell,
            lstm_layers=(256, 256),
            lstm_2_init_period=50,
            linear_layer_ref=noisy_linear,
            **kwargs
    ):
        super(GuidedPolicy_0_0, self).__init__(
            conv_2d_layer_config=conv_2d_layer_config,
            lstm_class_ref=lstm_class_ref,
            lstm_layers=lstm_layers,
            lstm_2_init_period=lstm_2_init_period,
            linear_layer_ref=linear_layer_ref,
            **kwargs
        )
        self.expert_actions = self.on_state_in['expert']


