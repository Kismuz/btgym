
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm as norm_layer

import numpy as np
import math

from btgym.algorithms.nn.layers import conv1d

def __conv_1d_casual_encoder(
        x,
        ob_space,
        ac_space,
        conv_1d_num_filters=32,
        conv_1d_filter_size=2,
        conv_1d_slice_size=1,
        conv_1d_activation=tf.nn.elu,
        conv_1d_use_bias=False,
        name='casual_encoder',
        reuse=False,
        collections=None,
        **kwargs
    ):
    """
    SLOOOOW
    Stage1 casual convolutions network: from 1D input to estimated features.

    Returns:
        tensor holding state features;
    """
    assert conv_1d_slice_size >=1

    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        shape = x.get_shape().as_list()
        if len(shape) > 3:  # remove pseudo 2d dimension
            x = x[:, :, 0, :]
        num_layers = int(np.log2(shape[1]))

        layer_stack = []
        # h_stack = [x[:, - conv_1d_slice_size, :]]
        y = x
        for i in range(num_layers):
            dilation = 2 ** i
            y = conv_1d_activation(
                norm_layer(
                    tf.layers.conv1d(
                        inputs=tf.pad(y, [[0, 0], [dilation, 0], [0, 0]]),
                        filters=conv_1d_num_filters,
                        kernel_size=conv_1d_filter_size,
                        strides=1,
                        padding='valid',
                        data_format='channels_last',
                        dilation_rate=dilation,
                        activation=None,
                        use_bias=conv_1d_use_bias,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name='conv1d_dialted_by_{}'.format(dilation),
                        # reuse=reuse
                    )
                )
            )
            # h_stack.append(y[:, - conv_1d_slice_size, :] ) - layer_stack[-1][:, - conv_1d_slice_size, :])
            layer_stack.append(y)

        encoded = tf.stack([h[:, - conv_1d_slice_size, :] for h in layer_stack], axis=1)

    return encoded


def conv_1d_casual_encoder(
        x,
        ob_space,
        ac_space,
        conv_1d_num_filters=32,
        conv_1d_filter_size=2,
        conv_1d_activation=tf.nn.elu,
        conv_1d_overlap=1,
        name='casual_encoder',
        reuse=False,
        collections=None,
        **kwargs
    ):
    """
    Tree-shaped convolution stack encoder as more comp. efficient alternative to dilated one.

    Stage1 casual convolutions network: from 1D input to estimated features.

    Returns:
        tensor holding state features;
    """

    with tf.variable_scope(name_or_scope=name, reuse=reuse):
        shape = x.get_shape().as_list()
        if len(shape) > 3:  # remove pseudo 2d dimension
            x = x[:, :, 0, :]
        num_layers = int(math.log(shape[1], conv_1d_filter_size))

        # print('num_layers: ', num_layers)

        layers = []
        slice_depth = []
        y = x

        for i in range(num_layers):

            _, length, channels = y.get_shape().as_list()

            # t2b:
            tail = length % conv_1d_filter_size
            if tail != 0:
                pad = conv_1d_filter_size - tail
                paddings = [[0, 0], [pad, 0], [0, 0]]
                y = tf.pad(y, paddings)
                length += pad

            # print('padded_length: ', length)

            num_time_batches = int(length / conv_1d_filter_size)

            # print('num_time_batches: ', num_time_batches)

            y = tf.reshape(y, [-1, conv_1d_filter_size, channels], name='layer_{}_t2b'.format(i))

            y = conv1d(
                x=y,
                num_filters=conv_1d_num_filters,
                filter_size=conv_1d_filter_size,
                stride=1,
                pad='VALID',
                name='conv1d_layer_{}'.format(i)
            )
            # b2t:
            y = tf.reshape(y, [-1, num_time_batches, conv_1d_num_filters], name='layer_{}_output'.format(i))

            y = norm_layer(y)
            if conv_1d_activation is not None:
                y = conv_1d_activation(y)

            layers.append(y)

            depth = conv_1d_overlap // conv_1d_filter_size ** i

            if depth < 1:
                depth = 1

            slice_depth.append(depth)

        # encoded = tf.stack([h[:, -1, :] for h in layers], axis=1, name='encoded_state')

        encoded = tf.concat(
            [
                tf.slice(
                    h,
                    begin=[0, h.get_shape().as_list()[1] - d, 0],
                    size=[-1, d, -1]
                ) for h, d in zip(layers, slice_depth)
            ],
            axis=1,
            name='encoded_state'
        )

        # encoded = tf.concat(layers, axis=1, name='full_encoded_state')
        # print('encoded :', encoded)

    return encoded

