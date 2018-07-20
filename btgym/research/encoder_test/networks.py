import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import layer_norm as norm_layer
from tensorflow.python.util.nest import flatten as flatten_nested

from btgym.algorithms.nn.layers import normalized_columns_initializer, categorical_sample
from btgym.algorithms.nn.layers import linear, noisy_linear, conv2d, deconv2d, conv1d
from btgym.algorithms.utils import rnn_placeholders


def conv_2d_network_skip(x,
                    ob_space,
                    ac_space,
                    conv_2d_layer_ref=conv2d,
                    conv_2d_num_filters=(32, 32, 64, 64),
                    conv_2d_filter_size=(3, 3),
                    conv_2d_stride=(2, 2),
                    pad="SAME",
                    dtype=tf.float32,
                    name='conv2d',
                    collections=None,
                    reuse=False,
                    keep_prob=None,
                    conv_2d_gated=True,
                    conv_2d_enable_skip=False,
                    **kwargs):
    """
    Convolution encoder with gated output

    Returns:
        tensor holding state features;
    """
    assert conv_2d_num_filters[-1] % 2 == 0
    layers = []
    with tf.variable_scope(name, reuse=reuse):
        for i, num_filters in enumerate(conv_2d_num_filters):
            x = tf.nn.elu(
                norm_layer(
                    conv_2d_layer_ref(
                        x,
                        num_filters,
                        "_layer_{}".format(i + 1),
                        conv_2d_filter_size,
                        conv_2d_stride,
                        pad,
                        dtype,
                        collections,
                        reuse
                    ),
                    scope=name + "_norm_layer_{}".format(i + 1)
                )
            )
            if keep_prob is not None:
                x = tf.nn.dropout(x, keep_prob=keep_prob, name="_layer_{}_with_dropout".format(i + 1))

            layers.append(x)

        if conv_2d_gated:
            x = layers.pop(-1)
            split_size = int(conv_2d_num_filters[-1] / 2)
            x1 = x[..., :split_size]
            x2 = x[..., split_size:]

            x = tf.multiply(
                x1,
                tf.nn.sigmoid(x2),
                name='gated_conv_output'
            )
            layers.append(x)

            # print('{}.shape = {}'.format(x.name, x.get_shape().as_list()))

        if conv_2d_enable_skip:
            x = tf.concat([tf.layers.flatten(l) for l in layers], axis=-1, name='flattened_encoded_state')

        # print('{}.shape = {}'.format(x.name, x.get_shape().as_list()))
        return x


def identity_encoder(x, name='identity_encoder', **kwargs):
    """
    Identity plug

    Returns:
        tensor holding state features;
    """
    with tf.variable_scope(name,):
        x = tf.layers.flatten(x)

        return x