# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Paper: https://arxiv.org/abs/1602.01783

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import flatten as batch_flatten
from tensorflow.python.util.nest import flatten as flatten_nested


class BaseLSTMPolicy(object):
    """
    Base policy estimator with multi-layer LSTM cells option.
    Input tensor `x_in` maps to LSTM layer.
    """
    x = None

    def __init__(self, x_in, ob_space, ac_space, lstm_class, lstm_layers):

        # Flatten end expand with fake time dim to feed to LSTM bank:
        x = tf.expand_dims(batch_flatten(x_in), [0])
        # x = tf.expand_dims(self.flatten_homebrew(x_in), [0])
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

        #print('GOT HERE 2, x:', x.shape)
        #print('GOT HERE 2, train_phase:', self.train_phase.shape)
        #print('GOT HERE 2, update_ops:', self.update_ops)

        # Define LSTM layers:
        lstm = []
        for size in lstm_layers:
            lstm += [lstm_class(size, state_is_tuple=True)]

        self.lstm = rnn.MultiRNNCell(lstm, state_is_tuple=True)
        # self.lstm = lstm[0]

        # Get time_dimension as [1]-shaped tensor:
        step_size = tf.expand_dims(tf.shape(x)[1], [0])
        #step_size = tf.shape(self.x)[:1]
        #print('GOT HERE 3')
        self.lstm_init_state = self.lstm.zero_state(1, dtype=tf.float32)

        lstm_state_pl = self.rnn_placeholders(self.lstm.zero_state(1, dtype=tf.float32))
        self.lstm_state_pl_flatten = flatten_nested(lstm_state_pl)

        #print('GOT HERE 4, x:', x.shape)
        lstm_outputs, self.lstm_state_out = tf.nn.dynamic_rnn(
            self.lstm,
            x,
            initial_state=lstm_state_pl,
            sequence_length=step_size,
            time_major=False
        )
        #print('GOT HERE 5')
        x = tf.reshape(lstm_outputs, [-1, lstm_layers[-1]])

        self.logits = self.linear(x, ac_space, "action", self.normalized_columns_initializer(0.01))
        self.vf = tf.reshape(self.linear(x, 1, "value", self.normalized_columns_initializer(1.0)), [-1])
        self.sample = self.categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # Add moving averages to save list (meant for Batch_norm layer):
        moving_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*moving.*')
        renorm_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*renorm.*')

        self.var_list += moving_var_list + renorm_var_list

    def get_initial_features(self):
        sess = tf.get_default_session()
        return sess.run(self.lstm_init_state)

    def flatten_homebrew(self, x):
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    def act(self, ob, lstm_state):
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update({self.x: [ob], self.train_phase: False})
        #print('#####_feeder:\n', feeder)
        return sess.run([self.sample, self.vf, self.lstm_state_out], feeder)

    def value(self, ob, lstm_state):
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update({self.x: [ob], self.train_phase: False})
        return sess.run(self.vf, feeder)[0]

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def linear(self, x, size, name, initializer=None, bias_init=0):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b

    def categorical_sample(self, logits, d):
        value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)

    def rnn_placeholders(self, state):
        """
        Converts RNN state tensors to placeholders .
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


class LSTMPolicy2D(BaseLSTMPolicy):
    """
    Policy estimator suited for Atari environments.
    """

    def __init__(self, ob_space, ac_space, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,)):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x_in_pl')

        # Conv layers:
        #for i in range(4):
        #    x = tf.nn.elu(self.conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        x = self._conv2d_constructor(x)

        super(LSTMPolicy2D, self).__init__(x, ob_space, ac_space, lstm_class, lstm_layers)

    def _conv2d_constructor(self,
                               x,
                               num_layers=4,
                               num_filters=32,
                               filter_size=(3, 3),
                               stride=(2, 2),
                               pad="SAME",
                               dtype=tf.float32,
                               collections=None,
                               reuse=False):
        """
        Defines graph of [possibly shared] 2d convolution network.
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


class LSTMPolicy(BaseLSTMPolicy):
    """
    Policy estimator directly feeds input to LSTM layer.
    """

    def __init__(self, ob_space, ac_space, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,)):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x_in_pl')

        super(LSTMPolicy, self).__init__(x, ob_space, ac_space, lstm_class, lstm_layers)


###############################################################################

class _LSTMPolicy_original(object):
    """
    Original Universe_Starter_Agent Policy Estimator. Kept here for reference.
    """
    def __init__(self, ob_space, ac_space):

        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.nn.elu(self.conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))


        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(self.flatten(x), [0])

        size = 256

        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        self.state_size = lstm.state_size
        # Get time_dimension as [1]-shaped tensor:
        step_size = tf.expand_dims(tf.shape(x)[1], [0])  # TODO: VERIFY!!!

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)

        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = self.linear(x, ac_space, "action", self.normalized_columns_initializer(0.01))
        self.vf = tf.reshape(self.linear(x, 1, "value", self.normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = self.categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def flatten(self, x):
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    def conv2d(self, x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
        with tf.variable_scope(name):
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

    def linear(self, x, size, name, initializer=None, bias_init=0):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b

    def categorical_sample(self, logits, d):
        value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)
