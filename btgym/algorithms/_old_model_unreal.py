# This UNREAL implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
# https://miyosuda.github.io/
# https://github.com/miyosuda/unreal
#
# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import flatten as batch_flatten
from tensorflow.python.util.nest import flatten as flatten_nested


class BaseUnrealPolicy(object):
    """
    Base CNN-LSTM policy estimator.
    """

    def __init__(self, ob_space, ac_space, rp_sequence_size, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,)):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class
        self.lstm_layers = lstm_layers

        # Placeholders for obs. state input:
        self.a3c_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='a3c_state_in_pl')
        self.off_a3c_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='off_policy_a3c_state_in_pl')
        self.rp_state_in = tf.placeholder(tf.float32, [rp_sequence_size-1] + list(ob_space), name='rp_state_in_pl')
        #self.vr_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='vr_state_in_pl')
        #self.pc_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='pc_state_in_pl')

        # Placeholders for concatenated action [one-hot] and reward [scalar]:
        self.a3c_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='a3c_action_reward_in_pl')
        self.off_a3c_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='off_policy_a3c_action_reward_in_pl')
        #self.vr_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='vr_action_reward_in_pl')
        #self.pc_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='pc_action_reward_in_pl')

        # Base on-policy A3C network:
        # Conv. layers:
        a3c_x = self._conv_2D_network_constructor(self.a3c_state_in, ob_space, ac_space)
        # LSTM layer takes conv. features and concatenated last action_reward tensor:
        [a3c_x, self.a3c_lstm_init_state, self.a3c_lstm_state_out, self.a3c_lstm_state_pl_flatten] =\
            self._lstm_network_constructor(a3c_x, self.a3c_a_r_in, lstm_class, lstm_layers, )
        # A3C policy and value outputs and action-sampling function:
        [self.a3c_logits, self.a3c_vf, self.a3c_sample] = self._dense_a3c_network_constructor(a3c_x, ac_space)

        # Off-policy A3C network (shared):
        off_a3c_x = self._conv_2D_network_constructor(self.off_a3c_state_in, ob_space, ac_space, reuse=True)
        [off_a3c_x_lstm_out, _, _, self.off_a3c_lstm_state_pl_flatten] =\
            self._lstm_network_constructor(off_a3c_x, self.off_a3c_a_r_in, lstm_class, lstm_layers, reuse=True)
        [self.off_a3c_logits, self.off_a3c_vf, _] =\
            self._dense_a3c_network_constructor(off_a3c_x_lstm_out, ac_space, reuse=True)

        # Aux1: `Pixel control` network:
        # Define pixels-change estimation function:
        # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
        [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
            self._pixel_change_2D_estimator_constructor(ob_space)

        #pc_x = self.conv_2d_network(self.pc_state_in, ob_space, ac_space, reuse=True)
        #[pc_x, _, _, self.pc_lstm_state_pl_flatten] =\
        #    self.lstm_network(pc_x, self.pc_a_r_in, lstm_class, lstm_layers, reuse=True)
        self.pc_state_in = self.off_a3c_state_in
        self.pc_a_r_in = self.off_a3c_a_r_in
        self.pc_lstm_state_pl_flatten = self.off_a3c_lstm_state_pl_flatten

        # Shared conv and lstm nets, same off-policy batch:
        pc_x = off_a3c_x_lstm_out

        # PC duelling Q-network, outputs [None, 20, 20, ac_size] Q-features tensor:
        self.pc_q = self._duelling_pc_network_constructor(pc_x)

        # Aux2: `Value function replay` network:
        # VR network is fully shared with a3c network but with `value` only output:
        # and has same off-policy batch pass with off_a3c network:

        #vr_x = self.conv_2d_network(self.vr_state_in, ob_space, ac_space, reuse=True)
        #[vr_x, _, _, self.vr_lstm_state_pl_flatten] =\
        #    self.lstm_network(vr_x, lstm_class, lstm_layers, reuse=True)
        self.vr_state_in = self.off_a3c_state_in
        self.vr_a_r_in = self.off_a3c_a_r_in
        #vr_x = off_a3c_x
        self.vr_lstm_state_pl_flatten = self.off_a3c_lstm_state_pl_flatten
        #[_, self.vr_value, _] = self._dense_a3c_network_constructor(vr_x, ac_space, reuse=True)
        self.vr_value = self.off_a3c_vf

        # Aux3: `Reward prediction` network:
        # Shared conv.:
        rp_x = self._conv_2D_network_constructor(self.rp_state_in, ob_space, ac_space, reuse=True)

        # RP output:
        self.rp_logits = self._dense_rp_network_constructor(rp_x)

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
        # Add moving averages to save list:
        moving_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*moving.*')
        renorm_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*renorm.*')

        # What to save:
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.var_list += moving_var_list + renorm_var_list

    def get_a3c_initial_features(self):
        """Called by thread-runner. Returns LSTM zero-state."""
        sess = tf.get_default_session()
        return sess.run(self.a3c_lstm_init_state)

    def flatten_homebrew(self, x):
        """Not used."""
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    def a3c_act(self, observation, lstm_state, action_reward):
        """Called by thread-runner."""
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.a3c_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update(
            {self.a3c_state_in: [observation],
             self.a3c_a_r_in: [action_reward],
             self.train_phase: False}
        )
        return sess.run([self.a3c_sample, self.a3c_vf, self.a3c_lstm_state_out], feeder)

    def get_a3c_value(self, observation, lstm_state, action_reward):
        """Called by thread-runner."""
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.a3c_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update(
            {self.a3c_state_in: [observation],
             self.a3c_a_r_in: [action_reward],
             self.train_phase: False}
        )
        return sess.run(self.a3c_vf, feeder)[0]

    def get_rp_prediction(self, ob, lstm_state):
        """Not used."""
        sess = tf.get_default_session()
        #feeder = {pl: value for pl, value in zip(self.rp_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        #feeder.update({self.rp_state_in: [ob], self.train_phase: False})
        feeder = {self.rp_state_in: [ob], self.train_phase: False}
        return sess.run(self.rp_logits, feeder)[0]

    def get_pc_target(self, state, last_state):
        """Called by thread-runner."""
        sess = tf.get_default_session()
        feeder = {self.pc_change_state_in: state, self.pc_change_last_state_in: last_state}
        return sess.run(self.pc_target, feeder)[0,...,0]

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
        """Called by thread-runner."""
        value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)

    def rnn_placeholders(self, state):
        """
        Given nested [multilayer] RNN state tensors, infers and returns state placeholders.
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

    def deconv2d(self, x, output_channels, name, filter_size=(4, 4), stride=(2, 2),
                 dtype=tf.float32, collections=None, reuse=False):
        """
        Deconvolutional layer, paper:
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
        Encapsulates convolutions, [possibly] skip-connections etc. Can be shared.
        Returns:
            output tensor;
        """
        for i in range(num_layers):
            x = tf.nn.elu(
                self.conv2d(x, num_filters, "conv2d_{}".format(i + 1), filter_size, stride, pad, dtype, collections, reuse)
            )
        # Following original paper design:
        #x = tf.nn.elu(self.conv2d(x, 16, 'conv2d_1', [8, 8], [4, 4], pad, dtype, collections, reuse))
        #x = tf.nn.elu(self.conv2d(x, 32, 'conv2d_2', [4, 4], [2, 2], pad, dtype, collections, reuse))
        #x = tf.nn.elu(
        #    self.linear(batch_flatten(x), 256, 'conv_2d_dense', self.normalized_columns_initializer(0.01), reuse=reuse)
        #)
        return x

    def _lstm_network_constructor(self, x, a_r, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,), reuse=False):
        """
        Stage2: from features to flattened LSTM output.
        Defines [multi-layered] dynamic [possibly] shared LSTM network.
        Returns:
             batch-wise flattened output tensor;
             lstm initial state tensor;
             lstm state output tensor;
             lstm flattened feed placeholders as tuple.
        """
        with tf.variable_scope('lstm', reuse=reuse):

            # Flatten, add action/reward and expand with fake time dim to feed LSTM bank:
            x = tf.concat([batch_flatten(x), a_r],axis=-1)
            x = tf.expand_dims(x, [0])

            # Define LSTM layers:
            lstm = []
            for size in lstm_layers:
                lstm += [lstm_class(size, state_is_tuple=True)]

            lstm = rnn.MultiRNNCell(lstm, state_is_tuple=True)
            # self.lstm = lstm[0]

            # Get time_dimension as [1]-shaped tensor:
            step_size = tf.expand_dims(tf.shape(x)[1], [0])

            lstm_init_state = lstm.zero_state(1, dtype=tf.float32)

            lstm_state_pl = self.rnn_placeholders(lstm.zero_state(1, dtype=tf.float32))
            lstm_state_pl_flatten = flatten_nested(lstm_state_pl)

            lstm_outputs, lstm_state_out = tf.nn.dynamic_rnn(
                lstm,
                x,
                initial_state=lstm_state_pl,
                sequence_length=step_size,
                time_major=False
            )
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

    def _pixel_change_2D_estimator_constructor(self, ob_space, stride=2):
        """
        Defines op for estimating `pixel change` as subsampled
        absolute difference of two states.
        """
        input_state = tf.placeholder(tf.float32, list(ob_space), name='pc_change_est_state_in')
        input_last_state = tf.placeholder(tf.float32, list(ob_space), name='pc_change_est_last_state_in')

        x = tf.abs(tf.subtract(input_state, input_last_state))
        x = tf.expand_dims(x, 0)[:, 1:-1, 1:-1, :]  # fake batch dim and crop
        x = tf.reduce_mean(x, axis=-1, keep_dims=True)
        # TODO: max_pool may be better?
        x_out = tf.nn.avg_pool(x, [1,stride,stride,1], [1,stride,stride,1], 'SAME')
        return input_state, input_last_state, x_out

    def _duelling_pc_network_constructor(self, x, reuse=False):
        """
        Stage3 network for `pixel control' task: from LSTM output to Q-aux. features.
        """
        x = tf.nn.elu(self.linear(x, 9*9*32, 'pc_dense', self.normalized_columns_initializer(0.01), reuse=reuse))
        x = tf.reshape(x, [-1, 9, 9, 32])
        pc_a = self.deconv2d(x, self.ac_space, 'pc_advantage', [4, 4], [2, 2], reuse=reuse) # [None, 20, 20, ac_size]
        pc_v = self.deconv2d(x, 1, 'pc_value_fn', [4, 4], [2, 2], reuse=reuse)  # [None, 20, 20, 1]

        # Q-value estimate using advantage mean,
        # see (9) in "Dueling Network Architectures..." paper:
        # https://arxiv.org/pdf/1511.06581.pdf
        pc_a_mean = tf.reduce_mean(pc_a, axis=3, keep_dims=True)
        pc_q = pc_v + pc_a - pc_a_mean  # [None, 20, 20, ac_size]

        return pc_q
