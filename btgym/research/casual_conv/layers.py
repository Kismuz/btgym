import tensorflow as tf

from btgym.algorithms.nn.layers import conv1d


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def dilated_conv1d(
        inputs,
        filters,
        filter_width=2,
        dilation_rate=1,
        pad='VALID',
        activation=None,
        name='dialted_conv_1d',
        reuse=False
):
    with tf.name_scope(name):
        if dilation_rate > 1:
            transformed = time_to_batch(inputs, dilation_rate)
            conv = conv1d(
                x=transformed,
                num_filters=filters,
                filter_size=filter_width,
                stride=1,
                pad=pad,
                name=name,
                reuse=reuse
            )
            restored = batch_to_time(conv, dilation_rate)
        else:
            restored = conv1d(
                x=inputs,
                num_filters=filters,
                filter_size=filter_width,
                stride=1,
                pad=pad,
                name=name,
                reuse=reuse
            )
        # Remove excess elements at the end.
        out_width = tf.shape(inputs)[1] - (filter_width - 1) * dilation_rate
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])

        if activation is not None:
            result = activation(result)

        return result

