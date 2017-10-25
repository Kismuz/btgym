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

from btgym.algorithms.model_util import *
import tensorflow as tf

class BasePpoPolicy(object):
    """
    Base CNN-LSTM policy estimator.
    """

    def __init__(self, ob_space, ac_space, rp_sequence_size, lstm_class=rnn.BasicLSTMCell, lstm_layers=(256,)):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class
        self.lstm_layers = lstm_layers
        self.callback = {}

        # Placeholders for obs. state input:
        self.on_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='on_policy_state_in_pl')
        self.off_state_in = tf.placeholder(tf.float32, [None] + list(ob_space), name='off_policy_state_in_pl')
        self.rp_state_in = tf.placeholder(tf.float32, [rp_sequence_size-1] + list(ob_space), name='rp_state_in_pl')

        # Placeholders for concatenated action [one-hot] and reward [scalar]:
        self.on_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='on_policy_action_reward_in_pl')
        self.off_a_r_in = tf.placeholder(tf.float32, [None, ac_space + 1], name='off_policy_action_reward_in_pl')
 
        # Base on-policy ppo network:
        # Conv. layers:
        ppo_x = conv_2d_network(self.on_state_in, ob_space, ac_space)
        # LSTM layer takes conv. features and concatenated last action_reward tensor:
        [ppo_x, self.on_lstm_init_state, self.on_lstm_state_out, self.on_lstm_state_pl_flatten] =\
            lstm_network(ppo_x, self.on_a_r_in, lstm_class, lstm_layers, )
        # ppo policy and value outputs and action-sampling function:
        [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(ppo_x, ac_space)

        # Off-policy ppo network (shared):
        off_ppo_x = conv_2d_network(self.off_state_in, ob_space, ac_space, reuse=True)
        [off_x_lstm_out, _, _, self.off_lstm_state_pl_flatten] =\
            lstm_network(off_ppo_x, self.off_a_r_in, lstm_class, lstm_layers, reuse=True)
        [self.off_ppo_logits, self.off_ppo_vf, _] =\
            dense_aac_network(off_x_lstm_out, ac_space, reuse=True)

        # Aux1: `Pixel control` network:
        # Define pixels-change estimation function:
        # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
        [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
            pixel_change_2d_estimator(ob_space)

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
        self.vr_value = self.off_ppo_vf

        # Aux3: `Reward prediction` network:
        # Shared conv.:
        rp_x = conv_2d_network(self.rp_state_in, ob_space, ac_space, reuse=True)

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
        self.callback['pixel_change'] = self._get_pc_target

    def get_initial_features(self):
        """Called by thread-runner. Returns LSTM zero-state."""
        sess = tf.get_default_session()
        return sess.run(self.on_lstm_init_state)

    def act(self, observation, lstm_state, action_reward):
        """Called by thread-runner."""
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.on_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update(
            {self.on_state_in: [observation],
             self.on_a_r_in: [action_reward],
             self.train_phase: False}
        )
        return sess.run([self.on_sample, self.on_vf, self.on_lstm_state_out], feeder)

    def get_value(self, observation, lstm_state, action_reward):
        """Called by thread-runner."""
        sess = tf.get_default_session()
        feeder = {pl: value for pl, value in zip(self.on_lstm_state_pl_flatten, flatten_nested(lstm_state))}
        feeder.update(
            {self.on_state_in: [observation],
             self.on_a_r_in: [action_reward],
             self.train_phase: False}
        )
        return sess.run(self.on_vf, feeder)[0]

    def _get_pc_target(self, state, last_state, **kwargs):
        """Called-back by thread-runner."""
        sess = tf.get_default_session()
        feeder = {self.pc_change_state_in: state, self.pc_change_last_state_in: last_state}
        return sess.run(self.pc_target, feeder)[0,...,0]

