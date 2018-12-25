
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
        keep_prob=None,
        conv_1d_gated=False,
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

            stride = conv_1d_filter_size - conv_1d_overlap
            assert stride > 0, 'conv_1d_filter_size should be greater then conv_1d_overlap'
        
            # print('num_time_batches: ', num_time_batches)

            y = tf.reshape(y, [-1, conv_1d_filter_size, channels], name='layer_{}_t2b'.format(i))

            y = conv1d(
                x=y,
                num_filters=conv_1d_num_filters,
                filter_size=conv_1d_filter_size,
                stride=stride,
                pad='VALID',
                name='conv1d_layer_{}'.format(i)
            )
            # b2t:
            y = tf.reshape(y, [-1, num_time_batches, conv_1d_num_filters], name='layer_{}_output'.format(i))

            y = norm_layer(y)
            if conv_1d_activation is not None:
                y = conv_1d_activation(y)

            if keep_prob is not None:
                y = tf.nn.dropout(y, keep_prob=keep_prob, name="_layer_{}_with_dropout".format(i))

            layers.append(y)

            depth = conv_1d_overlap // conv_1d_filter_size ** i

            if depth < 1:
                depth = 1

            slice_depth.append(depth)

        # encoded = tf.stack([h[:, -1, :] for h in layers], axis=1, name='encoded_state')

        sliced_layers = [
            tf.slice(
                h,
                begin=[0, h.get_shape().as_list()[1] - d, 0],
                size=[-1, d, -1]
            ) for h, d in zip(layers, slice_depth)
        ]
        output_stack = sliced_layers
        # But:
        if conv_1d_gated:
            split_size = int(conv_1d_num_filters / 2)
            output_stack = []
            for l in sliced_layers:
                x1 = l[..., :split_size]
                x2 = l[..., split_size:]

                y = tf.multiply(
                    x1,
                    tf.nn.sigmoid(x2),
                    name='gated_conv_output'
                )
                output_stack.append(y)

        encoded = tf.concat(output_stack, axis=1, name='encoded_state')

        # encoded = tf.concat(
        #     [
        #         tf.slice(
        #             h,
        #             begin=[0, h.get_shape().as_list()[1] - d, 0],
        #             size=[-1, d, -1]
        #         ) for h, d in zip(layers, slice_depth)
        #     ],
        #     axis=1,
        #     name='encoded_state'
        # )
        # print('encoder :', encoded)

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

    # alignments = attention_mechanism(query_state, None)  # normalized attention weights

    # Suppose there is no previous context for attention (uhm?):
    alignments = attention_mechanism(
        query_state,
        attention_mechanism.initial_alignments(tf.shape(inputs)[0], dtype=tf.float32)
    )
    # Somehow attention call returns tuple of twin tensors (wtf?):
    if isinstance(alignments, tuple):
        alignments = alignments[0]

    # Compute context vector:
    expanded_alignments = tf.expand_dims(alignments, axis=-2)

    # print('attention_mechanism.values:', attention_mechanism.values)
    # print('alignments: ', alignments)
    # print('expanded_alignments:', expanded_alignments)

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
        keep_prob=None,
        conv_1d_gated=False,
        conv_1d_full_hidden=False,
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

            if keep_prob is not None:
                y = tf.nn.dropout(y, keep_prob=keep_prob, name="_layer_{}_with_dropout".format(i))

            if conv_1d_gated:
                split_size = int(conv_1d_num_filters / 2)

                y1 = y[..., :split_size]
                y2 = y[..., split_size:]

                y = tf.multiply(
                    y1,
                    tf.nn.sigmoid(y2),
                    name="_layer_{}_gated".format(i)
                )
            layers.append(y)

            # Insert attention for all but top layer:
            if num_time_batches > 1:
                attention = attention_layer(
                    y,
                    attention_ref=conv_1d_attention_ref,
                    name='attention_layer_{}'.format(i)
                )
                attention_layers.append(attention)

        if conv_1d_full_hidden:
            convolved = tf.concat(layers, axis=-2, name='convolved_stack_full')

        else:
            convolved = tf.stack([h[:, -1, :] for h in layers], axis=1, name='convolved_stack')

        attended = tf.concat(attention_layers, axis=-2, name='attention_stack')

        encoded = tf.concat([convolved, attended], axis=-2, name='encoded_state')
        # print('layers', layers)
        # print('convolved: ', convolved)
        # print('attention_layers:', attention_layers)
        # print('attention_stack: ', attended)
        # print('encoded :', encoded)

    return encoded
