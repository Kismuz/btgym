
from btgym.algorithms.nn_utils import *
from btgym.algorithms.utils import *
import tensorflow as tf
from tensorflow.contrib.layers import flatten as batch_flatten

from btgym.algorithms import BaseAacPolicy, Aac1dPolicy


class GuideFFPolicy(BaseAacPolicy):

    def __init__(self,
                 ob_space,
                 ac_space,
                 ff_size=64,
                 **kwargs):
        """
        Simple and computationally cheap feed-forward policy.

        Args:
            ob_space:           dictionary of observation state shapes
            ac_space:           discrete action space shape (length)
            ff_size:            feed-forward dense layer size
            **kwargs            not used
        """
        kwargs.update(
            dict(
                conv_2d_filter_size=[3, 1],
                conv_2d_stride=[2, 1],

            )
        )

        self.ob_space = ob_space
        self.ac_space = ac_space
        self.aux_estimate = False
        self.callback = {}

        # Placeholders for obs. state input:
        self.on_state_in = nested_placeholders(ob_space, batch_dim=None, name='on_policy_state_in')
        self.off_state_in = nested_placeholders(ob_space, batch_dim=None, name='off_policy_state_in_pl')
        self.rp_state_in = nested_placeholders(ob_space, batch_dim=None, name='rp_state_in')

        # Placeholders for concatenated action [one-hot] and reward [scalar]:
        self.on_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='on_policy_action_reward_in_pl')
        self.off_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='off_policy_action_reward_in_pl')

        # Placeholders for rnn batch and time-step dimensions:
        self.on_batch_size = tf.placeholder(tf.int32, name='on_policy_batch_size')
        self.on_time_length = tf.placeholder(tf.int32, name='on_policy_sequence_size')

        self.off_batch_size = tf.placeholder(tf.int32, name='off_policy_batch_size')
        self.off_time_length = tf.placeholder(tf.int32, name='off_policy_sequence_size')

        # Base on-policy AAC network:
        # Conv. layers:
        on_aac_x = conv_2d_network(self.on_state_in['external'], ob_space['external'], ac_space, **kwargs)

        if False:
            # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
            x_shape_dynamic = tf.shape(on_aac_x)
            max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
            x_shape_static = on_aac_x.get_shape().as_list()

            on_a_r_in = tf.reshape(self.on_a_r_in, [self.on_batch_size, max_seq_len, ac_space + 1])
            on_aac_x = tf.reshape( on_aac_x, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

            # Feed last action_reward [, internal obs. state] into LSTM along with external state features:
            on_stage2_input = [on_aac_x, on_a_r_in]

            if 'internal' in list(self.on_state_in.keys()):
                x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
                x_int = tf.reshape(
                    self.on_state_in['internal'],
                    [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                on_stage2_input.append(x_int)

            on_aac_x = tf.concat(on_stage2_input, axis=-1)

        on_aac_x = batch_flatten(on_aac_x)

        # Dense layer:
        on_x_dense_out = tf.nn.elu(
            linear(on_aac_x, ff_size, 'dense_pi_v', normalized_columns_initializer(0.01), reuse=False)
        )

        # Dummy:
        self.on_lstm_init_state = (LSTMStateTuple(c=np.zeros((1,1)), h=np.zeros((1,1))),)
        self.on_lstm_state_out = (LSTMStateTuple(c=np.zeros((1,1)), h=np.zeros((1,1))),)
        self.on_lstm_state_pl_flatten = [
            tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dummy_c'),
            tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dummy_h')
        ]

        # Aac policy and value outputs and action-sampling function:
        [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(on_x_dense_out, ac_space)

        # Batch-norm related (useless, ignore):
        try:
            if self.train_phase is not None:
                pass

        except AttributeError:
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

    def get_initial_features(self, **kwargs):
        """
        Returns initial context.

        Returns:
            LSTM zero-state tuple.
        """
        return self.on_lstm_init_state

    def act(self, observation, lstm_state, action_reward):
        """
        Predicts action.

        Args:
            observation:    dictionary containing single observation
            action_reward:  concatenated last action-reward value

        Returns:
            Action [one-hot], V-fn value, output RNN state
        """
        sess = tf.get_default_session()
        feeder = feed_dict_from_nested(self.on_state_in, observation, expand_batch=True)
        feeder.update(
            {
                self.on_a_r_in: [action_reward],
                self.on_batch_size: 1,
                self.on_time_length: 1,
                self.train_phase: False
            }
        )

        sample = sess.run(self.on_sample, feeder)
        value = sess.run(self.on_vf, feeder)
        context = self.on_lstm_init_state
        return [sample, value, context]

    def get_value(self, observation, lstm_state, action_reward):
        """
        Estimates policy V-function.

        Args:
            observation:    single observation value
            action_reward:  concatenated last action-reward value

        Returns:
            V-function value
        """
        sess = tf.get_default_session()
        feeder = feed_dict_from_nested(self.on_state_in, observation, expand_batch=True)
        feeder.update(
            {
                self.on_a_r_in: [action_reward],
                self.on_batch_size: 1,
                self.on_time_length: 1,
                self.train_phase: False
            }
        )

        return sess.run(self.on_vf, feeder)[0]

