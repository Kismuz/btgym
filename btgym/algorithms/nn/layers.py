# Original code comes from OpenAI repository under MIT licence:
#
# https://github.com/openai/universe-starter-agent
# https://github.com/openai/baselines
#

import numpy as np
import tensorflow as tf


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# def categorical_sample(logits, d):
#     value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keepdims=True), 1), [1])
#     return tf.one_hot(value, d)

def categorical_sample(logits, depth):
    """
    Given logits returns one-hot encoded categorical sample.
    Args:
        logits:
        depth:

    Returns:
            tensor of shape [batch_dim, logits_depth]
    """
    # print('categorical_sample_logits: ', logits)
    value = tf.squeeze(tf.multinomial(logits, 1), [1])
    one_hot = tf.one_hot(value, depth, name='sample_one_hot')
    return one_hot


def linear(x, size, name, initializer=None, bias_init=0, reuse=False):
    """
    Linear network layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable("/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b


def noisy_linear(x, size, name, bias=True, activation_fn=tf.identity, reuse=False, **kwargs):
    """
    Noisy Net linear network layer using Factorised Gaussian noise;
    Code by Andrew Liao, https://github.com/andrewliao11/NoisyNet-DQN

    Papers:
        https://arxiv.org/abs/1706.10295
        https://arxiv.org/abs/1706.01905

    """
    with tf.variable_scope(name, reuse=reuse):
        # the function used in eq.7,8
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        # Initializer of \mu and \sigma
        mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),
                                                    maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
        sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
        # Sample noise from gaussian
        p = tf.random_normal([x.get_shape().as_list()[1], 1])
        q = tf.random_normal([1, size])
        f_p = f(p); f_q = f(q)
        w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable("/w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable("/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable("/b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable("/b_sigma", [size], initializer=sigma_init)
            b = b_mu + tf.multiply(b_sigma, b_epsilon)
            return activation_fn(ret + b)
        else:
            return activation_fn(ret)


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32,
           collections=None, reuse=False):
    """
    2D convolution layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        w = tf.get_variable("W", filter_shape, dtype, initializer=tf.contrib.layers.xavier_initializer(),
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

        w = tf.get_variable("d_W", filter_shape, dtype, initializer=tf.contrib.layers.xavier_initializer(),
                            collections=collections)
        b = tf.get_variable("d_b", [1, 1, 1, output_channels], initializer=tf.constant_initializer(0.0),
                            collections=collections)

        return tf.nn.conv2d_transpose(x, w, output_shape,
                                      strides=stride_shape,
                                      padding='VALID') + b


def conv1d(x, num_filters, name, filter_size=3, stride=2, pad="SAME", dtype=tf.float32,
           collections=None, reuse=False):
    """
    1D convolution layer.
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = stride

        # print('stride_shape:',stride_shape)

        filter_shape = [filter_size, int(x.get_shape()[-1]), num_filters]

        # print('filter_shape:', filter_shape)

        w = tf.get_variable("W", filter_shape, dtype, initializer=tf.contrib.layers.xavier_initializer(),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv1d(x, w, stride_shape, pad) + b


def conv2d_dw(x, num_filters, name='conv2d_dw', filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32,
              collections=None, reuse=False):
    """
    Depthwise 2D convolution layer. Slow, do not use.
    """
    with tf.variable_scope(name, reuse=reuse):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[-1]), num_filters]
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
                            tf.contrib.layers.xavier_initializer(), collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters * int(x.get_shape()[-1])],
                            initializer=tf.constant_initializer(0.0), collections=collections)
        return tf.nn.depthwise_conv2d(x, w, stride_shape, pad, [1, 1]) + b
