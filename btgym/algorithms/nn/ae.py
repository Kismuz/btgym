import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten as batch_flatten
from tensorflow.contrib.layers import layer_norm as norm_layer

from btgym.algorithms.nn.layers import normalized_columns_initializer, linear, conv2d


def conv2d_encoder(x,
                   layer_config=(
                        (32, (3, 1), (2, 1)),
                        (32, (3, 1), (2, 1)),
                        (32, (3, 1), (2, 1)),
                   ),
                   pad='SAME',
                   name='encoder',
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
    with tf.variable_scope(name, reuse=reuse):
        layer_shapes = [x.get_shape()]
        layer_outputs = []
        for i, layer_spec in enumerate(layer_config, 1):
            x = tf.nn.elu(
                norm_layer(
                    conv2d(
                        x=x,
                        num_filters=layer_spec[0],
                        name='/conv_kernels_{}'.format(i ),
                        filter_size=layer_spec[1],
                        stride=layer_spec[2],
                        pad=pad,
                        reuse=reuse
                    )
                ),
                name='encoder_layer_{}'.format(i),
            )
            layer_shapes.append(x.get_shape())
            layer_outputs.append(x)

        return layer_outputs, layer_shapes


def conv2d_decoder(z,
                   layer_shapes,
                   layer_config=(
                        (32, (3, 1), (2, 1)),
                        (32, (3, 1), (2, 1)),
                        (32, (3, 1), (2, 1)),
                   ),
                   pad='SAME',
                   resize_method=tf.image.ResizeMethod.BILINEAR,
                   name='decoder',
                   reuse=False):
    """
    Defines convolutional decoder.

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
    with tf.variable_scope(name, reuse=reuse):
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
                    name='conv_kernels_{}'.format(i),
                    filter_size=layer_spec[1],
                    stride=[1, 1],
                    pad=pad,
                    reuse=reuse
                ),
                name='decoder_layer_{}'.format(i),
            )
            layer_output.append(x)
        y_hat = conv2d(
            x=x,
            num_filters=layer_shapes[-1][-1],
            name='decoded_y_hat',
            filter_size=layer_config[-1][1],
            stride=[1, 1],
            pad='SAME',
            reuse=reuse
        )
        layer_output.append(y_hat)
        return layer_output


def conv2d_autoencoder(
        inputs,
        layer_config,
        resize_method=tf.image.ResizeMethod.BILINEAR,
        pad='SAME',
        linear_layer_ref=linear,
        name='base_conv2d_autoencoder',
        reuse=False,
        **kwargs
    ):
    """
    Basic convolutional autoencoder.
    Hidden state is passed through dense linear layer.

    Args:
        inputs:             input tensor
        layer_config:       layers configuration list: [layer_1_config, layer_2_config,...], where:
                            layer_i_config = [num_filters(int), filter_size(list), stride(list)];
                            this list represent decoder part of autoencoder bottleneck,
                            decoder part is inferred symmetrically
        resize_method:      up-sampling method, one of supported tf.image.ResizeMethod's
        pad:                str, padding scheme: 'SAME' or 'VALID'
        linear_layer_ref:   linear layer class to use
        name:               str, mame scope
        reuse:              bool

    Returns:
        list of tensors holding encoded features, layer_wise from outer to inner
        tensor holding batch-wise flattened hidden state vector
        list of tensors holding decoded features, layer-wise from inner to outer
        tensor holding reconstructed output
        None value

    """
    with tf.variable_scope(name, reuse=reuse):
        # Encode:
        encoder_layers, shapes = conv2d_encoder(
            x=inputs,
            layer_config=layer_config,
            pad=pad,
            reuse=reuse
        )
        # Flatten hidden state, pass through dense :
        z = batch_flatten(encoder_layers[-1])
        h, w, c = encoder_layers[-1].get_shape().as_list()[1:]

        z = linear_layer_ref(
            x=z,
            size=h * w * c,
            name='hidden_dense',
            initializer=normalized_columns_initializer(1.0),
            reuse=reuse
        )
        # Reshape back and feed to decoder:
        decoder_layers = conv2d_decoder(
            z=tf.reshape(z, [-1, h, w, c]),
            layer_config=layer_config,
            layer_shapes=shapes,
            pad=pad,
            resize_method=resize_method,
            reuse=reuse
        )
        y_hat = decoder_layers[-1]
        return encoder_layers, z, decoder_layers, y_hat, None


def cw_conv2d_autoencoder(
        inputs,
        layer_config,
        resize_method=tf.image.ResizeMethod.BILINEAR,
        pad='SAME',
        linear_layer_ref=linear,
        name='cw_conv2d_autoencoder',
        reuse=False,
        **kwargs
    ):
    """
    Channel-wise convolutional autoencoder.
    Hidden state is passed through dense linear layer.
    Pain-slow, do not use.

    Args:
        inputs:             input tensor
        layer_config:       layers configuration list: [layer_1_config, layer_2_config,...], where:
                            layer_i_config = [num_filters(int), filter_size(list), stride(list)];
                            this list represent decoder part of autoencoder bottleneck,
                            decoder part is inferred symmetrically
        resize_method:      up-sampling method, one of supported tf.image.ResizeMethod's
        pad:                str, padding scheme: 'SAME' or 'VALID'
        linear_layer_ref:   linear layer class to use
        name:               str, mame scope
        reuse:              bool

    Returns:
        per-channel list of lists of tensors holding encoded features, layer_wise from outer to inner
        tensor holding batch-wise flattened hidden state vector
        per-channel list of lists of tensors holding decoded features, layer-wise from inner to outer
        tensor holding reconstructed output
        None value

    """
    with tf.variable_scope(name, reuse=reuse):
        ae_bank = []
        for i in range(inputs.get_shape().as_list()[-1]):
            # Making list of list of AE's:
            encoder_layers, z, decoder_layers, y_hat, _ = conv2d_autoencoder(
                inputs=inputs[..., i][..., None],
                layer_config=layer_config,
                resize_method=resize_method,
                linear_layer_ref=linear_layer_ref,
                name='ae_channel_{}'.format(i),
                pad=pad
            )
            ae = dict(
                inputs=inputs[..., i][..., None],
                encoder_layers=encoder_layers,
                z=z,
                decoder_layers=decoder_layers,
                y_hat=y_hat,
            )

            ae_bank.append(ae)

        y_hat = []
        z = []
        cw_encoder_layers = []
        cw_decoder_layers = []

        for ae in ae_bank:
            y_hat.append(ae['y_hat'])
            z.append(ae['z'])
            cw_encoder_layers.append(ae['encoder_layers'])
            cw_decoder_layers.append(ae['decoder_layers'])

        # Flatten hidden state:
        z = tf.concat(z, axis=-1, name='hidden_state')

        # encoder_layers = []
        # for layer in zip(*cw_encoder_layers):
        #     encoder_layers.append(tf.concat(layer, axis=-2))
        #
        # decoder_layers = []
        # for layer in zip(*cw_decoder_layers):
        #     decoder_layers.append(tf.concat(layer, axis=-2))

        # Reshape back reconstruction:
        y_hat = tf.concat(y_hat, axis=-1, name='decoded_y_hat')

        return cw_encoder_layers, z, cw_decoder_layers, y_hat, None


def beta_var_conv2d_autoencoder(
        inputs,
        layer_config,
        resize_method=tf.image.ResizeMethod.BILINEAR,
        pad='SAME',
        linear_layer_ref=linear,
        name='vae_conv2d',
        max_batch_size=256,
        reuse=False
    ):
    """
    Variational autoencoder.

    Papers:
        https://arxiv.org/pdf/1312.6114.pdf
        https://arxiv.org/pdf/1606.05908.pdf
        http://www.matthey.me/pdf/betavae_iclr_2017.pdf


    Args:
        inputs:             input tensor
        layer_config:       layers configuration list: [layer_1_config, layer_2_config,...], where:
                            layer_i_config = [num_filters(int), filter_size(list), stride(list)];
                            this list represent decoder part of autoencoder bottleneck,
                            decoder part is inferred symmetrically
        resize_method:      up-sampling method, one of supported tf.image.ResizeMethod's
        pad:                str, padding scheme: 'SAME' or 'VALID'
        linear_layer_ref:   linear layer class - not used
        name:               str, mame scope
        max_batch_size:     int, dynamic batch size should be no greater than this value
        reuse:              bool

    Returns:
        list of tensors holding encoded features, layer_wise from outer to inner
        tensor holding batch-wise flattened hidden state vector
        list of tensors holding decoded features, layer-wise from inner to outer
        tensor holding reconstructed output
        tensor holding estimated KL divergence

    """
    with tf.variable_scope(name, reuse=reuse):

        # Encode:
        encoder_layers, shapes = conv2d_encoder(
            x=inputs,
            layer_config=layer_config,
            pad=pad,
            reuse=reuse
        )
        # Flatten hidden state, pass through dense:
        z_flat = batch_flatten(encoder_layers[-1])

        h, w, c = encoder_layers[-1].get_shape().as_list()[1:]

        z = tf.nn.elu(
            linear(
                x=z_flat,
                size=h * w * c,
                name='enc_dense',
                initializer=normalized_columns_initializer(1.0),
                reuse=reuse
            )
        )
        # TODO: revert back to dubled Z-size
        # half_size_z = h * w * c
        # size_z = 2 * half_size_z

        size_z = int(h * w * c/2)
        z = tf.nn.elu(
            linear(
                #x=z_flat,
                x=z,
                #size=size_z,
                size=size_z * 2,
                name='hidden_dense',
                initializer=normalized_columns_initializer(1.0),
                reuse=reuse
            )
        )
        # Get sample parameters:
        #mu, log_sigma = tf.split(z, [half_size_z, half_size_z], axis=-1)
        mu, log_sigma = tf.split(z, [size_z, size_z], axis=-1)

        # Oversized noise generator:
        #eps = tf.random_normal(shape=[max_batch_size, half_size_z], mean=0., stddev=1.)
        eps = tf.random_normal(shape=[max_batch_size, size_z], mean=0., stddev=1.)
        eps = eps[:tf.shape(z)[0],:]

        # Get sample z ~ Q(z|X):
        z_sampled = mu + tf.exp(log_sigma / 2) * eps

        # D_KL(Q(z|X) || P(z|X)):
        # TODO: where is sum?!
        d_kl = 0.5 * (tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma)

        # Reshape back and feed to decoder:

        z_sampled_dec = tf.nn.elu(
            linear(
                x=z_sampled,
                size=h * w * c,
                name='dec_dense',
                initializer=normalized_columns_initializer(1.0),
                reuse=reuse
            )
        )

        decoder_layers = conv2d_decoder(
            z=tf.reshape(z_sampled_dec, [-1, h, w, c]),
            layer_config=layer_config,
            layer_shapes=shapes,
            pad=pad,
            resize_method=resize_method,
            reuse=reuse
        )
        y_hat = decoder_layers[-1]
        return encoder_layers, z_sampled, decoder_layers, y_hat, d_kl


class KernelMonitor():
    """
    Visualises convolution filters learnt for specific layer.
    Source: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    """

    def __init__(self, conv_input, layer_output):
        """

        Args:
            conv_input:         convolution stack input tensor
            layer_output:       tensor holding output of layer of interest from stack
        """
        self.idx = tf.placeholder(tf.int32, name='kernel_index')
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
