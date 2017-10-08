# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Paper: https://arxiv.org/abs/1602.01783

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import flatten as batch_flatten
from tensorflow.python.util.nest import flatten as flatten_nested


class BaseUnrealPolicy(object):
    """
    Base policy estimator with multi-layer LSTM cells option.
    """
    #x = None
    #a3c_state_in = None
    #rp_state_in = None

    def __init__(self, ob_space, ac_space, rp_sequence_size, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,)):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class
        self.lstm_layers = lstm_layers

        # Placeholders:
        self.a3c_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='a3c_state_in_pl')
        self.rp_state_in = tf.placeholder(tf.float32, [rp_sequence_size-1] + list(ob_space), name='rp_state_in_pl')
        self.vr_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='vr_state_in_pl')

        # Batch-norm related (useless, ignore):
        try:
            if self.train_phase is not None:
                pass

        except:
            self.train_phase = tf.placeholder_with_default(
                tf.constant(False, dtype=tf.bool),
                shape=(),
                name='train_phase_flag_pl'
            )
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Define base A3C policy network:
        # Conv layers:
        a3c_x = self._conv_2D_network_constructor(self.a3c_state_in, ob_space, ac_space)
        # LSTM layers:
        [a3c_x, self.a3c_lstm_init_state, self.a3c_lstm_state_out, self.a3c_lstm_state_pl_flatten] =\
            self._lstm_network_constructor(a3c_x, lstm_class, lstm_layers, )
        # A3C specific:
        [self.a3c_logits, self.a3c_vf, self.a3c_sample] = self._dense_a3c_network_constructor(a3c_x, ac_space)

        # Pixel control network:


        # Value fn. replay network:
        # If I got it correct, vr network is fully shared with a3c net but with `value` only output:
        vr_x = self._conv_2D_network_constructor(self.vr_state_in, ob_space, ac_space, reuse=True)

        [vr_x, _, _, self.vr_lstm_state_pl_flatten] =\
            self._lstm_network_constructor(vr_x, lstm_class, lstm_layers, reuse=True)

        [_, self.vr_value, _] = self._dense_a3c_network_constructor(vr_x, ac_space, reuse=True)


        # Reward prediction network:
        # Shared conv.:
        rp_x = self._conv_2D_network_constructor(self.rp_state_in, ob_space, ac_space, reuse=True)
        # Shared LSTM: ???? - no LSTM, just flatten tensor and put it in softmax!
        #[rp_x, _, _, self.rp_lstm_state_pl_flatten] =\
        #    self._lstm_network_constructor(rp_x, lstm_class, lstm_layers, reuse=True)

        # RP output:
        self.rp_logits = self._dense_rp_network_constructor(rp_x)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # Add moving averages to save list (meant for Batch_norm layer):
        moving_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*moving.*')
        renorm_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*renorm.*')

        self.var_list += moving_var_list + renorm_var_list

    def get_a3c_initial_features(self):
        sess = tf.get_default_session()
        return sess.run(self.a3c_lstm_init_state)

    def flatten_homebrew(self, x):
        """Not used."""
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    def a3c_act(self, ob, lstm_state):
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.a3c_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update({self.a3c_state_in: [ob], self.train_phase: False})
        #print('#####_feeder:\n', feeder)
        return sess.run([self.a3c_sample, self.a3c_vf, self.a3c_lstm_state_out], feeder)

    def get_a3c_value(self, ob, lstm_state):
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.a3c_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update({self.a3c_state_in: [ob], self.train_phase: False})
        return sess.run(self.a3c_vf, feeder)[0]

    def get_rp_prediction(self, ob, lstm_state):
        """Test one, not used at train time."""
        sess = tf.get_default_session()
        #feeder = {pl: value for pl, value in zip(self.rp_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        #feeder.update({self.rp_state_in: [ob], self.train_phase: False})
        feeder = {self.rp_state_in: [ob], self.train_phase: False}
        return sess.run(self.rp_logits, feeder)[0]

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def linear(self, x, size, name, initializer=None, bias_init=0, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("/w", [x.get_shape()[1], size], initializer=initializer)
            b = tf.get_variable("/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b

    def categorical_sample(self, logits, d):
        value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)

    def rnn_placeholders(self, state):
        """
        Converts RNN state tensors to placeholders.
        """
        if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
            c, h = state
            c = tf.placeholder(tf.float32, c.shape, c.op.name + '_c_pl')
            h = tf.placeholder(tf.float32, h.shape, h.op.name + '_h_pl')
            return tf.contrib.rnn.LSTMStateTuple(c, h)
        elif isinstance(state, tf.Tensor):
            h = state
            h = tf.placeholder(tf.float32, h.shape, h.op.name + '_h_pl')
            return h
        else:
            structure = [self.rnn_placeholders(x) for x in state]
            return tuple(structure)

    def conv2d(self, x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32,
               collections=None, reuse=False):
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

    def _conv_2D_network_constructor(self,
                                     x,
                                     ob_space,
                                     ac_space,
                                     num_layers=4,
                                     num_filters=32,
                                     filter_size=(3, 3),
                                     stride=(2, 2),
                                     pad="SAME",
                                     dtype=tf.float32,
                                     collections=None,
                                     reuse=False):
        """
        Stage1 network: from preprocessed 2D input to estimated features.
        Encapsulates convolutions, [possibly] skip-connections etc. [Possibly] shared.
        Returns:
            output tensor;
        """
        for i in range(num_layers):
            x = tf.nn.elu(
                self.conv2d(
                    x,
                    num_filters,
                    "conv2d_{}".format(i + 1),
                    filter_size,
                    stride,
                    pad,
                    dtype,
                    collections,
                    reuse
                )
            )
        return x

    def _lstm_network_constructor(self, x, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,), reuse=False):
        """
        Stage2: from features to flattened LSTM output.
        Defines [multi-layered] dynamic [possibly] shared LSTM network.
        Returns:
             batch-wise flattened output tensor;
             lstm initial state tensor;
             lstm state output tensor;
             lstm flattened feed placeholder tensor;
        """
        with tf.variable_scope('lstm', reuse=reuse):
            # Flatten and expand with fake time dim to feed to LSTM bank:
            x = tf.expand_dims(batch_flatten(x), [0])

            # Define LSTM layers:
            lstm = []
            for size in lstm_layers:
                lstm += [lstm_class(size, state_is_tuple=True)]

            lstm = rnn.MultiRNNCell(lstm, state_is_tuple=True)
            # self.lstm = lstm[0]

            # Get time_dimension as [1]-shaped tensor:
            step_size = tf.expand_dims(tf.shape(x)[1], [0])
            #step_size = tf.shape(self.x)[:1]
            #print('GOT HERE 3')
            lstm_init_state = lstm.zero_state(1, dtype=tf.float32)

            lstm_state_pl = self.rnn_placeholders(lstm.zero_state(1, dtype=tf.float32))
            lstm_state_pl_flatten = flatten_nested(lstm_state_pl)

            #print('GOT HERE 4, x:', x.shape)
            lstm_outputs, lstm_state_out = tf.nn.dynamic_rnn(
                lstm,
                x,
                initial_state=lstm_state_pl,
                sequence_length=step_size,
                time_major=False
            )
            #print('GOT HERE 5')
            x_out = tf.reshape(lstm_outputs, [-1, lstm_layers[-1]])
        return x_out, lstm_init_state, lstm_state_out, lstm_state_pl_flatten

    def _dense_a3c_network_constructor(self, x, ac_space, reuse=False):
        """
        Stage3: from LSTM flattened output to a3c-specifc values.
        Returns: A3C logits, value function and action sampling function.
        """
        logits = self.linear(x, ac_space, "action", self.normalized_columns_initializer(0.01), reuse=reuse)
        vf = tf.reshape(self.linear(x, 1, "value", self.normalized_columns_initializer(1.0), reuse=reuse), [-1])
        sample = self.categorical_sample(logits, ac_space)[0, :]

        return logits, vf, sample

    def _dense_rp_network_constructor(self, x):
        """
        Stage3: From shared convolutions to reward-prediction task output.
        """
        #print('x_shape:', x.get_shape())
        x = tf.reshape(x, [1, -1]) # flatten to pretend we got batch of size 1
        # Fully connected x128 followed by 3-way classifier [with softmax], as in paper,
        x = tf.nn.elu(self.linear(x, 128, 'rp_dense', self.normalized_columns_initializer(0.01)))
        #print('x_shape2:', x.get_shape())
        logits = self.linear(x, 3, 'rp_classifier', self.normalized_columns_initializer(0.01))
        # Note:  softmax is actually not here but inside loss operation (see unreal.py)
        return logits
