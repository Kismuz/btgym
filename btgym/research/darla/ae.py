import tensorflow as tf
from tensorflow.contrib.layers import layer_norm as norm_layer
from btgym.algorithms.nn_utils import conv2d
import numpy as np


def conv2d_encoder(x,
                   layer_config=(
                        (64, (2, 1), (2, 1)),
                        (32, (2, 1), (2, 1)),
                        (16, (2, 1), (2, 1)),
                   ),
                   pad='SAME',
                   name='conv2d_encoder',
                   reuse=False):
    """
    Defines convolutional encoder.

    Args:
        x:              input tensor
        layer_config:   first to last nested layers configuration list: [layer_1_config, layer_2_config,...], where:
                        layer_i_config = [num_filters(int), filter_size(list), stride(list)]
        pad:            str, padding scheme: 'SAME' or 'VALID'
        name:           str, mame scope
        reuse:          bool

    Returns:
        list of tensors holding encoded features for every layer outer to inner,
        level-wise list of encoding layers shapes, first ro last.

    """
    layer_shapes = [x.get_shape()]
    layer_outputs = []
    for i, layer_spec in enumerate(layer_config, 1):
        x = tf.nn.elu(
            norm_layer(
                conv2d(
                    x=x,
                    num_filters=layer_spec[0],
                    name=name+'/conv_enc_layer_{}'.format(i ),
                    filter_size=layer_spec[1],
                    stride=layer_spec[2],
                    pad=pad,
                    reuse=reuse
                )
            ),
            name=name + '/encoder_layer_{}'.format(i),
        )
        layer_shapes.append(x.get_shape())
        layer_outputs.append(x)

    return layer_outputs, layer_shapes


def conv2d_decoder(z,
                   layer_shapes,
                   layer_config=(
                        (64, (2, 1), (2, 1)),
                        (32, (2, 1), (2, 1)),
                        (16, (2, 1), (2, 1)),
                   ),
                   pad='SAME',
                   resize_method=tf.image.ResizeMethod.BILINEAR,
                   name='conv2d_decoder',
                   reuse=False):
    """
    Builds convolutional decoder.

    Args:
        z:                  tensor holding encoded state
        layer_shapes:       level-wise list of matching encoding layers shapes, last to first.
        layer_config:       layers configuration list: [layer_1_config, layer_2_config,...], where:
                            layer_i_config = [num_filters(int), filter_size(list), stride(list)]
        pad:                str, padding scheme: 'SAME' or 'VALID'
        resize_method:      up-sampling method, one of supported tf.image.ResizeMethod's
        name:               str, mame scope
        reuse:              bool

    Returns:
        list of tensors holding decoded features for every layer inner to outer

    """
    x = z
    layer_shapes = list(layer_shapes)
    layer_shapes.reverse()
    layer_config = list(layer_config)
    layer_config.reverse()
    layer_output = []
    for i, (layer_spec, layer_shape) in enumerate(zip(layer_config,layer_shapes[1:]), 1):
        x = tf.image.resize_images(
            images=x,
            size=[int(layer_shape[1]), int(layer_shape[2])],
            method=resize_method,
        )
        x = tf.nn.elu(
            conv2d(
                x=x,
                num_filters=layer_spec[0],
                name=name + '/conv_dec_layer_{}'.format(i),
                filter_size=layer_spec[1],
                stride=[1, 1],
                pad=pad,
                reuse=reuse
            ),
            name=name + '/decoder_layer_{}'.format(i),
        )
        layer_output.append(x)
    y_hat = conv2d(
        x=x,
        num_filters=layer_shapes[-1][-1],
        name=name + '/decoded_y_hat',
        filter_size=layer_config[-1][1],
        stride=[1, 1],
        pad='SAME',
        reuse=reuse
    )
    layer_output.append(y_hat)
    return layer_output


def conv2d_autoencoder(inputs,
                       layer_config,
                       resize_method=tf.image.ResizeMethod.BILINEAR,
                       pad='SAME',
                       name='base_conv2d_autoencoder',
                       reuse=False):
    """
    Basic convolutional autoencoder.

    Args:
        inputs:         input tensor
        layer_config:   layers configuration list: [layer_1_config, layer_2_config,...], where:
                        layer_i_config = [num_filters(int), filter_size(list), stride(list)];
                        this list represent decoder part of autoencoder bottleneck,
                        decoder part is inferred symmetrically
        pad:            str, padding scheme: 'SAME' or 'VALID'
        name:           str, mame scope
        reuse:          bool

    Returns:
        tensor holding encoded features, layer_wise from outer to inner
        tensor holding decoded geatures, layer-wise from inner to outer

    """
    encoder_layers, shapes = conv2d_encoder(
        x=inputs,
        layer_config=layer_config,
        pad=pad,
        name=name,
        reuse=reuse
    )
    decoder_layers = conv2d_decoder(
        z=encoder_layers[-1],
        layer_config=layer_config,
        layer_shapes=shapes,
        pad=pad,
        resize_method=resize_method,
        name=name,
        reuse=reuse
    )
    return encoder_layers, decoder_layers


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

