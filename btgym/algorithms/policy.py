# Asynchronous implementation of Proximal Policy Optimization algorithm.
# paper:
# https://arxiv.org/pdf/1707.06347.pdf
#
# Based on PPO-SGD code from OpenAI `Baselines` repository under MIT licence:
# https://github.com/openai/baselines
#
# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#

from btgym.algorithms.nnet_util import *
from btgym.algorithms.util import *
import tensorflow as tf


class BaseAacPolicy(object):
    """
    Base advantage actor-critic LSTM policy estimator with auxiliary control tasks.

    Papers:
    https://arxiv.org/abs/1602.01783

    https://arxiv.org/abs/1611.05397
    """

    def __init__(self, ob_space, ac_space, rp_sequence_size,
                 lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,), aux_estimate=True, **kwargs):
        """
        Defines [partially shared] on/off-policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects uni-modal observation as array of shape `ob_space`.

        Args:
            ob_space:           dictionary of observation state shapes
            ac_space:           discrete action space shape (length)
            rp_sequence_size:   reward prediction sample length
            lstm_class:         tf.nn.lstm class
            lstm_layers:        tuple of LSTM layers sizes
            aux_estimate:       (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary.
            **kwargs            not used
        """
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class
        self.lstm_layers = lstm_layers
        self.aux_estimate = aux_estimate
        self.callback = {}

        # Placeholders for obs. state input:
        self.on_state_in = nested_placeholders(ob_space, batch_dim=None, name='on_policy_state_in')
        self.off_state_in = nested_placeholders(ob_space, batch_dim=None, name='off_policy_state_in_pl')
        self.rp_state_in = nested_placeholders(ob_space, batch_dim=rp_sequence_size-1, name='rp_state_in')

        # Placeholders for concatenated action [one-hot] and reward [scalar]:
        self.on_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='on_policy_action_reward_in_pl')
        self.off_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='off_policy_action_reward_in_pl')

        # Base on-policy AAC network:
        # Conv. layers:
        on_aac_x = conv_2d_network(self.on_state_in['external'], ob_space['external'], ac_space)
        # LSTM layer takes conv. features and concatenated last action_reward tensor:
        [on_aac_x, self.on_lstm_init_state, self.on_lstm_state_out, self.on_lstm_state_pl_flatten] =\
            lstm_network(on_aac_x, self.on_a_r_in, lstm_class, lstm_layers, )
        # aac policy and value outputs and action-sampling function:
        [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(on_aac_x, ac_space)

        # Off-policy AAC network (shared):
        off_aac_x = conv_2d_network(self.off_state_in['external'], ob_space['external'], ac_space, reuse=True)
        [off_x_lstm_out, _, _, self.off_lstm_state_pl_flatten] =\
            lstm_network(off_aac_x, self.off_a_r_in, lstm_class, lstm_layers, reuse=True)
        [self.off_logits, self.off_vf, _] =\
            dense_aac_network(off_x_lstm_out, ac_space, reuse=True)

        # Aux1: `Pixel control` network:
        # Define pixels-change estimation function:
        # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
        [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
            pixel_change_2d_estimator(ob_space['external'])

        self.pc_state_in = self.off_state_in
        self.pc_a_r_in = self.off_a_r_in
        self.pc_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten

        # Shared conv and lstm nets, same off-policy batch:
        pc_x = off_x_lstm_out

        # PC duelling Q-network, outputs [None, 20, 20, ac_size] Q-features tensor:
        self.pc_q = duelling_pc_network(pc_x, self.ac_space)

        # Aux2: `Value function replay` network:
        # VR network is fully shared with ppo network but with `value` only output:
        # and has same off-policy batch pass with off_ppo network:

        self.vr_state_in = self.off_state_in
        self.vr_a_r_in = self.off_a_r_in

        self.vr_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten
        self.vr_value = self.off_vf

        # Aux3: `Reward prediction` network:
        # Shared conv.:
        rp_x = conv_2d_network(self.rp_state_in['external'], ob_space['external'], ac_space, reuse=True)

        # RP output:
        self.rp_logits = dense_rp_network(rp_x)

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

        # Callbacks:
        if self.aux_estimate:
            self.callback['pixel_change'] = self.get_pc_target

    def get_initial_features(self):
        """Returns initial context.

        Returns:
            LSTM zero-state tuple.
        """
        sess = tf.get_default_session()
        return sess.run(self.on_lstm_init_state)

    def act(self, observation, lstm_state, action_reward):
        """
        Predicts action.

        Args:
            observation:    dictionary containing single observation
            lstm_state:     lstm context value
            action_reward:  concatenated last action-reward value

        Returns:
            Action, one-hot.
        """
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.on_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update(feed_dict_from_nested(self.on_state_in, observation, expand_batch=True))
        feeder.update(
            {#self.on_state_in: [observation],
             self.on_a_r_in: [action_reward],
             self.train_phase: False}
        )
        return sess.run([self.on_sample, self.on_vf, self.on_lstm_state_out], feeder)

    def get_value(self, observation, lstm_state, action_reward):
        """
        Estimates policy V-function.

        Args:
            observation:    single observation value
            lstm_state:     lstm context value
            action_reward:  concatenated last action-reward value

        Returns:
            Policy V-function estimated value
        """
        sess = tf.get_default_session()
        feeder = feed_dict_rnn_context(self.on_lstm_state_pl_flatten, lstm_state)
        feeder.update(feed_dict_from_nested(self.on_state_in, observation, expand_batch=True))
        feeder.update({self.on_a_r_in: [action_reward], self.train_phase: False})

        return sess.run(self.on_vf, feeder)[0]

    def get_pc_target(self, state, last_state, **kwargs):
        """
        Estimates pixel-control task target.

        Args:
            state:      single observation value
            last_state: single observation value
            **kwargs:   not used

        Returns:
            Estimated absolute difference between two subsampled states.
        """
        sess = tf.get_default_session()
        feeder = {self.pc_change_state_in: state['external'], self.pc_change_last_state_in: last_state['external']}

        return sess.run(self.pc_target, feeder)[0,...,0]

