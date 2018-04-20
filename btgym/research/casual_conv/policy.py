import tensorflow as tf

from btgym.research.gps.policy import GuidedPolicy_0_0
from btgym.research.casual_conv.networks import conv_1d_casual_encoder


class CasualConvPolicy_0_0(GuidedPolicy_0_0):
    """
    Casual.0.
    """
    def __init__(
        self,
        state_encoder_class_ref=conv_1d_casual_encoder,
        conv_1d_num_filters=32,
        conv_1d_filter_size=2,
        conv_1d_slice_size=1,  # future use, do not modify yet
        conv_1d_activation=tf.nn.elu,
        conv_1d_use_bias=False,
        **kwargs
    ):
        assert conv_1d_slice_size == 1

        super().__init__(
            state_encoder_class_ref=state_encoder_class_ref,
            conv_1d_num_filters=conv_1d_num_filters,
            conv_1d_filter_size=conv_1d_filter_size,
            conv_1d_slice_size=conv_1d_slice_size,
            conv_1d_activation=conv_1d_activation,
            conv_1d_use_bias=conv_1d_use_bias,
            **kwargs
        )

