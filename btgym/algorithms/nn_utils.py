# Original code comes from OpenAI repository under MIT licence:
#
# https://github.com/openai/universe-starter-agent
# https://github.com/openai/baselines
#

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import flatten as batch_flatten
from tensorflow.python.util.nest import flatten as flatten_nested
from btgym.algorithms.utils import rnn_placeholders


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def linear(x, size, name, initializer=None, bias_init=0, reuse=False):
    """
    Linear network layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable("/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32,
           collections=None, reuse=False):
    """
    2D convolution layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def deconv2d(x, output_channels, name, filter_size=(4, 4), stride=(2, 2),
             dtype=tf.float32, collections=None, reuse=False):
    """
    Deconvolution layer, paper:
    http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = [1, stride[0], stride[1], 1]

        batch_size = tf.shape(x)[0]
        input_height = int(x.get_shape()[1])
        input_width = int(x.get_shape()[2])
        input_channels = int(x.get_shape()[3])

        out_height = (input_height - 1) * stride[0] + filter_size[0]
        out_width = (input_width - 1) * stride[1] + filter_size[1]

        filter_shape = [filter_size[0], filter_size[1], output_channels, input_channels]
        output_shape = tf.stack([batch_size, out_height, out_width, output_channels])

        fan_in = np.prod(filter_shape[:2]) * input_channels
        fan_out = np.prod(filter_shape[:2]) * output_channels
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("d_W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("d_b", [1, 1, 1, output_channels], initializer=tf.constant_initializer(0.0),
                            collections=collections)

        return tf.nn.conv2d_transpose(x, w, output_shape,
                                      strides=stride_shape,
                                      padding='VALID') + b


def conv1d(x, num_filters, name, filter_size=3, stride=2, pad="SAME", dtype=tf.float32,
           collections=None, reuse=False):
    """
    1D convolution layer
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = stride

        # print('stride_shape:',stride_shape)

        filter_shape = [filter_size, int(x.get_shape()[-1]), num_filters]

        # print('filter_shape:', filter_shape)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:2])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:1]) * num_filters

        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv1d(x, w, stride_shape, pad) + b


def conv_2d_network(x,
                    ob_space,
                    ac_space,
                    conv_2d_num_layers=4,
                    conv_2d_num_filters=32,
                    conv_2d_filter_size=(3, 3),
                    conv_2d_stride=(2, 2),
                    pad="SAME",
                    dtype=tf.float32,
                    collections=None,
                    reuse=False,
                    **kwargs):
    """
    Stage1 network: from preprocessed 2D input to estimated features.
    Encapsulates convolutions, [possibly] skip-connections etc. Can be shared.

    Returns:
        tensor holding state features;
    """
    for i in range(conv_2d_num_layers):
        x = tf.nn.elu(
            conv2d(
                x,
                conv_2d_num_filters,
                "conv2d_{}".format(i + 1),
                conv_2d_filter_size,
                conv_2d_stride,
                pad,
                dtype,
                collections,
                reuse
            )
        )
    # A3c/BaseAAC original paper design:
    # x = tf.nn.elu(conv2d(x, 16, 'conv2d_1', [8, 8], [4, 4], pad, dtype, collections, reuse))
    # x = tf.nn.elu(conv2d(x, 32, 'conv2d_2', [4, 4], [2, 2], pad, dtype, collections, reuse))
    # x = tf.nn.elu(
    #    linear(batch_flatten(x), 256, 'conv_2d_dense', self.normalized_columns_initializer(0.01), reuse=reuse)
    # )
    return x


def conv_1d_network(x,
                    ob_space,
                    ac_space,
                    conv_1d_num_layers=4,
                    conv_1d_num_filters=32,
                    conv_1d_filter_size=3,
                    conv_1d_stride=2,
                    pad="SAME",
                    dtype=tf.float32,
                    collections=None,
                    reuse=False):
    """
    Stage1 network: from preprocessed 1D input to estimated features.
    Encapsulates convolutions, [possibly] skip-connections etc. Can be shared.

    Returns:
        tensor holding state features;
    """
    for i in range(conv_1d_num_layers):
        x = tf.nn.elu(
            conv1d(
                x,
                conv_1d_num_filters,
                "conv1d_{}".format(i + 1),
                conv_1d_filter_size,
                conv_1d_stride,
                pad,
                dtype,
                collections,
                reuse
            )
        )
    return x


def lstm_network(x, lstm_sequence_length, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,), reuse=False, **kwargs):
    """
    Stage2 network: from features to flattened LSTM output.
    Defines [multi-layered] dynamic [possibly shared] LSTM network.

    Returns:
         batch-wise flattened output tensor;
         lstm initial state tensor;
         lstm state output tensor;
         lstm flattened feed placeholders as tuple.
    """
    with tf.variable_scope('lstm', reuse=reuse):
        # Flatten, add action/reward and expand with fake [time] batch? dim to feed LSTM bank:
        #x = tf.concat([x, a_r] ,axis=-1)
        #x = tf.concat([batch_flatten(x), a_r], axis=-1)
        #x = tf.expand_dims(x, [0])

        # Define LSTM layers:
        lstm = []
        for size in lstm_layers:
            lstm += [lstm_class(size, state_is_tuple=True)]

        lstm = rnn.MultiRNNCell(lstm, state_is_tuple=True)
        # Get time_dimension as [1]-shaped tensor:
        step_size = tf.expand_dims(tf.shape(x)[1], [0])

        lstm_init_state = lstm.zero_state(1, dtype=tf.float32)

        lstm_state_pl = rnn_placeholders(lstm.zero_state(1, dtype=tf.float32))
        lstm_state_pl_flatten = flatten_nested(lstm_state_pl)

        lstm_outputs, lstm_state_out = tf.nn.dynamic_rnn(
            lstm,
            x,
            initial_state=lstm_state_pl,
            sequence_length=lstm_sequence_length,
            time_major=False
        )
        #x_out = tf.reshape(lstm_outputs, [-1, lstm_layers[-1]])
        x_out = lstm_outputs
    return x_out, lstm_init_state, lstm_state_out, lstm_state_pl_flatten


def dense_aac_network(x, ac_space, reuse=False):
    """
    Stage3 network: from LSTM flattened output to advantage actor-critic.

    Returns:
        logits tensor
        value function tensor
        action sampling function.
    """
    logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01), reuse=reuse)
    vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0), reuse=reuse), [-1])
    sample = categorical_sample(logits, ac_space)[0, :]

    return logits, vf, sample


def dense_rp_network(x):
    """
    Stage3 network: From shared convolutions to reward-prediction task output tensor.
    """
    # print('x_shape:', x.get_shape())
    #x = tf.reshape(x, [1, -1]) # flatten to pretend we got batch of size 1

    # Fully connected x128 followed by 3-way classifier [with softmax], as in paper:
    x = tf.nn.elu(linear(x, 128, 'rp_dense', normalized_columns_initializer(0.01)))
    # print('x_shape2:', x.get_shape())
    logits = linear(x, 3, 'rp_classifier', normalized_columns_initializer(0.01))
    # Note:  softmax is actually not here but inside loss operation (see losses.py)
    return logits


def pixel_change_2d_estimator(ob_space, pc_estimator_stride=(2, 2), **kwargs):
    """
    Defines tf operation for estimating `pixel change` as subsampled absolute difference of two states.

    Note:
        crops input array by one pix from either side; --> 1D signal to be shaped as [signal_length, 3]
    """
    input_state = tf.placeholder(tf.float32, list(ob_space), name='pc_change_est_state_in')
    input_last_state = tf.placeholder(tf.float32, list(ob_space), name='pc_change_est_last_state_in')

    x = tf.abs(tf.subtract(input_state, input_last_state)) # TODO: tf.square?

    if x.shape[-2] <= 3:
        x = tf.expand_dims(x, 0)[:, 1:-1, :, :]  # Assume 1D signal, fake batch dim and crop H dim only
        #x = tf.transpose(x, perm=[0, 1, 3, 2])  # Swap channels and height for
    else:
        x = tf.expand_dims(x, 0)[:, 1:-1, 1:-1, :]  # True 2D,  fake batch dim and crop H, W dims

    x = tf.reduce_mean(x, axis=-1, keep_dims=True)

    x_out = tf.nn.max_pool(
        x,
        [1, pc_estimator_stride[0], pc_estimator_stride[1], 1],
        [1, pc_estimator_stride[0], pc_estimator_stride[1], 1],
        'SAME'
    )
    return input_state, input_last_state, x_out


def duelling_pc_network(x,
                        ac_space,
                        duell_pc_x_inner_shape=(9, 9, 32),
                        duell_pc_filter_size=(4, 4),
                        duell_pc_stride=(2, 2),
                        reuse=False,
                        **kwargs):
    """
    Stage3 network for `pixel control' task: from LSTM output to Q-aux. features tensor.
    """
    x = tf.nn.elu(
        linear(x, np.prod(duell_pc_x_inner_shape), 'pc_dense', normalized_columns_initializer(0.01), reuse=reuse)
    )
    x = tf.reshape(x, [-1] + list(duell_pc_x_inner_shape))
    pc_a = deconv2d(x, ac_space, 'pc_advantage', duell_pc_filter_size, duell_pc_stride, reuse=reuse)  # [None, 20, 20, ac_size]
    pc_v = deconv2d(x, 1, 'pc_value_fn', duell_pc_filter_size, duell_pc_stride, reuse=reuse)  # [None, 20, 20, 1]

    # Q-value estimate using advantage mean,
    # as (9) in "Dueling Network Architectures..." paper:
    # https://arxiv.org/pdf/1511.06581.pdf
    pc_a_mean = tf.reduce_mean(pc_a, axis=-1, keep_dims=True)
    pc_q = pc_v + pc_a - pc_a_mean  # [None, 20, 20, ac_size]

    return pc_q

