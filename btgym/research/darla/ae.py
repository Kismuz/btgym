import tensorflow as tf
from tensorflow.contrib.layers import layer_norm as norm_layer
from btgym.algorithms.nn_utils import conv2d
import numpy as np


def conv2d_encoder(x,
                   filters=(64, 32, 16),
                   filter_size=(2, 1),
                   stride=(2, 1),
                   pad='SAME',
                   name='conv2d_encoder',
                   reuse=False):
    """
    Defines convolutional encoder.

    Args:
        x:              input tensor
        filters:        list, number of conv. kernels in each layer
        filter_size:    list, conv. kernel size
        stride:         list, stride size
        pad:            str, padding scheme: 'SAME' or 'VALID'
        name:           str, mame scope
        reuse:          bool

    Returns:
        tensor holding encoded features,
        level-wise list of encoding layers shapes, first ro last.

    """
    layer_shapes = [x.get_shape()]
    for i in range(len(filters)):
        x = tf.nn.elu(
            norm_layer(
                conv2d(
                    x=x,
                    num_filters=filters[i],
                    name=name+'/encoder_layer_{}'.format(i + 1),
                    filter_size=filter_size,
                    stride=stride,
                    pad=pad,
                    reuse=reuse
                )
            )
        )
        layer_shapes.append(x.get_shape())

    return x, layer_shapes


def conv2d_decoder(z,
                   layer_shapes,
                   filters=(64, 32, 16),
                   filter_size=(2, 1),
                   pad='SAME',
                   resize_method=tf.image.ResizeMethod.BILINEAR,
                   name='conv2d_decoder',
                   reuse=False):
    """
    Builds convolutional decoder.

    Args:
        z:                  tensor holding encoded state
        layer_shapes:       level-wise list of matching encoding layers shapes, last to first.
        filters:            list, number of conv. kernels in each layer
        filter_size:
        filter_size:        list, conv. kernel size
        pad:                str, padding scheme: 'SAME' or 'VALID'
        resize_method:      up-sampling method, one of supported tf.image.ResizeMethod's
        name:               str, mame scope
        reuse:              bool

    Returns:
        tensor holding decoded value

    """
    x = z
    for i in range(len(filters)):
        x = tf.image.resize_images(
            images=x,
            size=[int(layer_shapes[-2 - i][1]), int(layer_shapes[-2 - i][2])],
            method=resize_method,
        )
        x = tf.nn.elu(
            conv2d(
                x=x,
                num_filters=filters[-1 - i],
                name=name + '/decoder_layer_{}'.format(i + 1),
                filter_size=filter_size,
                stride=[1, 1],
                pad=pad,
                reuse=reuse
            )
        )
    y_hat = conv2d(
        x=x,
        num_filters=layer_shapes[0][-1],
        name=name + '/decoded_y_hat',
        filter_size=filter_size,
        stride=[1, 1],
        pad='SAME',
        reuse=reuse
    )
    return y_hat


def conv2d_autoencoder(inputs,
                       filters=(64, 32, 16),
                       filter_size=(2, 1),
                       stride=(2, 1),
                       pad='SAME',
                       name='base_conv2d_autoencoder',
                       reuse=False):
    """
    Basic convolutional autoencoder.

    Args:
        inputs:         input tensor
        filters:        list, number of conv. kernels in each layer
        filter_size:    list, conv. kernel size
        stride:         list, stride size
        pad:            str, padding scheme: 'SAME' or 'VALID'
        name:           str, mame scope
        reuse:          bool

    Returns:
        tensor holding encoded features,
        tensor holding input approximation

    """
    z, shapes = conv2d_encoder(
        x=inputs,
        filters=filters,
        filter_size=filter_size,
        stride=stride,
        pad=pad,
        name=name,
        reuse=reuse
    )
    y_hat = conv2d_decoder(
        z=z,
        layer_shapes=shapes,
        filters=filters,
        filter_size=filter_size,
        pad=pad,
        name=name,
        reuse=reuse
    )
    return z, y_hat


class KernelMonitor():
    """
    Visualises convolution filters for specific layer in convolution stack.
    Source: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    """

    def __init__(self, conv_input, layer_output):
        """

        Args:
            conv_input:         convolution stack input tensor
            layer_output:       tensor holding output of layer of interest from stack
        """
        self.idx = tf.placeholder(tf.int32, name='kernel_index')  # can be any integer from 0 to 15
        self.conv_input = conv_input
        self.layer_output = layer_output
        # Build a loss function that maximizes the activation
        # of the n-th filter of the layer considered:
        self.vis_loss = tf.reduce_mean(self.layer_output[:, :, :, self.idx])

        # Gradient of the input picture wrt this loss:
        self.vis_grads = tf.gradients(self.vis_loss, self.conv_input)[0]

        # Normalization trick:
        self.vis_grads /= (tf.sqrt(tf.reduce_mean(tf.square(self.vis_grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    def _iterate(self, sess, signal, kernel_index):
        """
        Returns the loss and grads for specified kernel given the input signal

        Args:
            sess:           tf.Session object
            signal:         input signal to convolution stack
            kernel_index:   filter number in layer considered

        Returns:
            loss and gradients values
        """
        return sess.run([self.vis_loss, self.vis_grads], {self.conv_input: signal, self.idx: kernel_index})

    def fit(self, sess, kernel_index, step=1e-3, num_steps=40):
        """
        Learns input signal that maximizes the activation of given kernel.

        Args:
            sess:               tf.Session object
            kernel_index:       filter number of interest
            step:               gradient ascent step size
            num_steps:          number of steps to fit

        Returns:
            learnt signal as np.array

        """
        # Start from some noise:
        signal = np.random.random([1] + self.conv_input.get_shape().as_list()[1:])

        # Run gradient ascent:
        for i in range(num_steps):
            loss_value, grads_value = self._iterate(sess, signal, kernel_index)
            signal += grads_value * step

        return signal

