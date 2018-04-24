
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm as norm_layer

import numpy as np
import math

from btgym.algorithms.nn.layers import conv1d


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


def attention_layer(inputs, attention_ref=tf.contrib.seq2seq.LuongAttention, name='attention_layer', **kwargs):
    """
    Temporal attention layer.
    Computes attention context based on last(left) value in time dim.

    Paper:
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.,
    "Effective Approaches to Attention-based Neural Machine Translation." https://arxiv.org/abs/1508.04025

    Args:
        inputs:
        attention_ref:      attention mechanism class
        name:

    Returns:
        attention output tensor
    """
    shape = inputs.get_shape().as_list()
    source_states = inputs[:, :-1, :]  # all but last
    query_state = inputs[:, -1, :]

    attention_mechanism = attention_ref(
        num_units=shape[-1],
        memory=source_states,
        #scale=True,
        name=name,
        **kwargs
    )

    alignments = attention_mechanism(query_state, None)  # normalized attention weights

    # Compute context vector:
    expanded_alignments = tf.expand_dims(alignments, 1)

    context = tf.matmul(expanded_alignments, attention_mechanism.values)  # values == source_states

    # context = tf.squeeze(context, [1])
    # attention = tf.layers.Dense(shape-1, name='attention_layer')(tf.concat([query_state, context], 1))

    attention = context

    return attention


def conv_1d_casual_attention_encoder(
        x,
        ob_space,
        ac_space,
        conv_1d_num_filters=32,
        conv_1d_filter_size=2,
        conv_1d_activation=tf.nn.elu,
        conv_1d_attention_ref=tf.contrib.seq2seq.LuongAttention,
        name='casual_encoder',
        reuse=False,
        collections=None,
        **kwargs
    ):
    """
    Tree-shaped convolution stack encoder with self-attention.

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
        attention_layers = []
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

            # Insert attention for all but top layer:
            if num_time_batches > 1:
                attention = attention_layer(
                    y,
                    attention_ref=conv_1d_attention_ref,
                    name='attention_layer_{}'.format(i)
                )
                attention_layers.append(attention)

        convolved = tf.stack([h[:, -1, :] for h in layers], axis=1, name='convolved_stack')
        attended = tf.concat(attention_layers, axis=-2, name='attention_stack')

        encoded = tf.concat([convolved, attended], axis=-2, name='encoded_state')

        # print('convolved: ', convolved)
        # print('attention_stack: ', attended)
        # print('encoded :', encoded)

    return encoded
