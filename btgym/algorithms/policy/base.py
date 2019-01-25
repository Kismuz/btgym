# Asynchronous implementation of a3c/ppo algorithm.
# paper:
# https://arxiv.org/pdf/1707.06347.pdf
#
# Based on PPO-SGD code from OpenAI `Baselines` repository under MIT licence:
# https://github.com/openai/baselines
#
# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#

from gym.spaces import Discrete, Dict

from btgym.algorithms.nn.networks import *
from btgym.algorithms.utils import *
from btgym.algorithms.math_utils import sample_dp, softmax
from btgym.datafeed.base import EnvResetConfig
from btgym.spaces import DictSpace, ActionDictSpace
from gym.spaces import Discrete


class BaseAacPolicy(object):
    """
    Base advantage actor-critic Convolution-LSTM policy estimator with auxiliary control tasks for
    discrete or nested discrete action spaces.

    Papers:

        https://arxiv.org/abs/1602.01783
        https://arxiv.org/abs/1611.05397
    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size,
                 lstm_class=rnn.BasicLSTMCell,
                 lstm_layers=(256,),
                 action_dp_alpha=200.0,
                 aux_estimate=False,
                 **kwargs):
        """
        Defines [partially shared] on/off-policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects multi-modal observation as array of shape `ob_space`.

        Args:
            ob_space:           instance of btgym.spaces.DictSpace
            ac_space:           instance of btgym.spaces.ActionDictSpace
            rp_sequence_size:   reward prediction sample length
            lstm_class:         tf.nn.lstm class
            lstm_layers:        tuple of LSTM layers sizes
            aux_estimate:       bool, if True - add auxiliary tasks estimations to self.callbacks dictionary
            time_flat:          bool, if True - use static rnn, dynamic otherwise
            **kwargs            not used
        """
        assert isinstance(ob_space, DictSpace), \
            'Expected observation space be instance of btgym.spaces.DictSpace, got: {}'.format(ob_space)
        self.ob_space = ob_space

        assert isinstance(ac_space, ActionDictSpace), \
            'Expected action space be instance of btgym.spaces.ActionDictSpace, got: {}'.format(ac_space)

        assert ac_space.base_space == Discrete, \
            'Base policy restricted to gym.spaces.Discrete base action spaces, got: {}'.format(ac_space.base_space)

        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class
        self.lstm_layers = lstm_layers
        self.action_dp_alpha = action_dp_alpha
        self.aux_estimate = aux_estimate
        self.callback = {}

        # Placeholders for obs. state input:
        self.on_state_in = nested_placeholders(self.ob_space.shape, batch_dim=None, name='on_policy_state_in')
        self.off_state_in = nested_placeholders(self.ob_space.shape, batch_dim=None, name='off_policy_state_in_pl')
        self.rp_state_in = nested_placeholders(self.ob_space.shape, batch_dim=None, name='rp_state_in')

        # Placeholders for previous step action[multi-categorical vector encoding]  and reward [scalar]:
        self.on_last_a_in = tf.placeholder(
            tf.float32,
            [None, self.ac_space.encoded_depth],
            name='on_policy_last_action_in_pl'
        )
        self.on_last_reward_in = tf.placeholder(tf.float32, [None], name='on_policy_last_reward_in_pl')

        self.off_last_a_in = tf.placeholder(
            tf.float32,
            [None, self.ac_space.encoded_depth],
            name='off_policy_last_action_in_pl'
        )
        self.off_last_reward_in = tf.placeholder(tf.float32, [None], name='off_policy_last_reward_in_pl')

        # Placeholders for rnn batch and time-step dimensions:
        self.on_batch_size = tf.placeholder(tf.int32, name='on_policy_batch_size')
        self.on_time_length = tf.placeholder(tf.int32, name='on_policy_sequence_size')

        self.off_batch_size = tf.placeholder(tf.int32, name='off_policy_batch_size')
        self.off_time_length = tf.placeholder(tf.int32, name='off_policy_sequence_size')

        try:
            if self.train_phase is not None:
                pass

        except AttributeError:
            self.train_phase = tf.placeholder_with_default(
                tf.constant(False, dtype=tf.bool),
                shape=(),
                name='train_phase_flag_pl'
            )
        # Base on-policy AAC network:
        # Conv. layers:
        on_aac_x = conv_2d_network(self.on_state_in['external'], self.ob_space.shape['external'], ac_space, **kwargs)

        # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(on_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
        x_shape_static = on_aac_x.get_shape().as_list()

        on_last_action_in = tf.reshape(
            self.on_last_a_in,
            [self.on_batch_size, max_seq_len, self.ac_space.encoded_depth]
        )
        on_r_in = tf.reshape(self.on_last_reward_in, [self.on_batch_size, max_seq_len, 1])

        on_aac_x = tf.reshape( on_aac_x, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # print('*** POLICY DEBUG ***')
        # print('self.on_last_a_in :', self.on_last_a_in)
        # print('on_last_action_in: ', on_last_action_in)
        # print('on_r_in: ', on_r_in)
        # print('on_aac_x: ', on_aac_x)

        # Feed last action, reward [, internal obs. state] into LSTM along with external state features:
        on_stage2_input = [on_aac_x, on_last_action_in, on_r_in]

        if 'internal' in list(self.on_state_in.keys()):
            x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
            x_int = tf.reshape(
                self.on_state_in['internal'],
                [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
            )
            on_stage2_input.append(x_int)

        on_aac_x = tf.concat(on_stage2_input, axis=-1)

        # print('on_stage2_input->on_aac_x: ', on_aac_x)

        # LSTM layer takes conv. features and concatenated last action_reward tensor:
        [on_x_lstm_out, self.on_lstm_init_state, self.on_lstm_state_out, self.on_lstm_state_pl_flatten] =\
            lstm_network(
                x=on_aac_x,
                lstm_sequence_length=self.on_time_length,
                lstm_class=lstm_class,
                lstm_layers=lstm_layers,
            )

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_out.get_shape().as_list()
        on_x_lstm_out = tf.reshape(on_x_lstm_out, [x_shape_dynamic[0], x_shape_static[-1]])

        # Aac policy and value outputs and action-sampling function:
        [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(on_x_lstm_out, self.ac_space.one_hot_depth)

        # Off-policy AAC network (shared):
        off_aac_x = conv_2d_network(self.off_state_in['external'], self.ob_space.shape['external'], ac_space, reuse=True, **kwargs)

        # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(off_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.off_batch_size, tf.int32)
        x_shape_static = off_aac_x.get_shape().as_list()

        off_action_in = tf.reshape(
            self.off_last_a_in,
            [self.off_batch_size, max_seq_len, self.ac_space.encoded_depth]
        )
        off_r_in = tf.reshape(self.off_last_reward_in, [self.off_batch_size, max_seq_len, 1])  # reward is scalar

        off_aac_x = tf.reshape( off_aac_x, [self.off_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        off_stage2_input = [off_aac_x, off_action_in, off_r_in]

        if 'internal' in list(self.off_state_in.keys()):
            x_int_shape_static = self.off_state_in['internal'].get_shape().as_list()
            off_x_int = tf.reshape(
                self.off_state_in['internal'],
                [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
            )
            off_stage2_input.append(off_x_int)

        off_aac_x = tf.concat(off_stage2_input, axis=-1)

        [off_x_lstm_out, _, _, self.off_lstm_state_pl_flatten] =\
            lstm_network(off_aac_x, self.off_time_length, lstm_class, lstm_layers, reuse=True)

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = off_x_lstm_out.get_shape().as_list()
        off_x_lstm_out = tf.reshape(off_x_lstm_out, [x_shape_dynamic[0], x_shape_static[-1]])

        # Off policy dense:
        [self.off_logits, self.off_vf, _] = dense_aac_network(off_x_lstm_out, self.ac_space.one_hot_depth, reuse=True)

        # Aux1: `Pixel control` network:
        # Define pixels-change estimation function:
        # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
        [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
            pixel_change_2d_estimator(self.ob_space.shape['external'], **kwargs)

        self.pc_batch_size = self.off_batch_size
        self.pc_time_length = self.off_time_length

        self.pc_state_in = self.off_state_in
        #self.pc_a_r_in = self.off_last_action_in
        self.pc_last_a_in = self.off_last_a_in
        self.pc_last_reward_in = self.off_last_reward_in
        self.pc_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten

        # Shared conv and lstm nets, same off-policy batch:
        pc_x = off_x_lstm_out

        # PC duelling Q-network, outputs [None, 20, 20, ac_size] Q-features tensor:
        # Restricted to single action space:
        act_space = self.ac_space.one_hot_depth
        self.pc_q = duelling_pc_network(pc_x, act_space, **kwargs)

        # Aux2: `Value function replay` network:
        # VR network is fully shared with ppo network but with `value` only output:
        # and has same off-policy batch pass with off_ppo network:
        self.vr_batch_size = self.off_batch_size
        self.vr_time_length = self.off_time_length

        self.vr_state_in = self.off_state_in
        #self.vr_a_r_in = self.off_last_action_in
        self.vr_last_a_in = self.off_last_a_in
        self.vr_last_reward_in = self.off_last_reward_in

        self.vr_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten
        self.vr_value = self.off_vf

        # Aux3: `Reward prediction` network:
        self.rp_batch_size = tf.placeholder(tf.int32, name='rp_batch_size')

        # Shared conv. output:
        rp_x = conv_2d_network(self.rp_state_in['external'], self.ob_space.shape['external'], ac_space, reuse=True, **kwargs)

        # Flatten batch-wise:
        rp_x_shape_static = rp_x.get_shape().as_list()
        rp_x = tf.reshape(rp_x, [self.rp_batch_size, np.prod(rp_x_shape_static[1:]) * (self.rp_sequence_size-1)])

        # RP output:
        self.rp_logits = dense_rp_network(rp_x)

        # Batch-norm related :
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

    def get_initial_features(self, **kwargs):
        """
        Returns initial context.

        Returns:
            LSTM zero-state tuple.
        """
        # TODO: rework as in: AacStackedMetaPolicy --> base runner, verbose runner; synchro_runner ok
        sess = tf.get_default_session()
        return sess.run(self.on_lstm_init_state)

    def act(self, observation, lstm_state, last_action, last_reward, deterministic=False):
        """
        Emits action.

        Args:
            observation:    dictionary containing single observation
            lstm_state:     lstm context value
            last_action:    action value from previous step
            last_reward:    reward value previous step
            deterministic:  bool, it True - act deterministically,
                            use random sampling otherwise (default);
                            effective for discrete action sapce only (TODO: continious)

        Returns:
            Action as dictionary of several action encodings, actions logits, V-fn value, output RNN state
        """
        try:
            sess = tf.get_default_session()
            feeder = {pl: value for pl, value in zip(self.on_lstm_state_pl_flatten, flatten_nested(lstm_state))}
            feeder.update(feed_dict_from_nested(self.on_state_in, observation, expand_batch=True))
            feeder.update(
                {
                    self.on_last_a_in: last_action,
                    self.on_last_reward_in: last_reward,
                    self.on_batch_size: 1,
                    self.on_time_length: 1,
                    self.train_phase: False
                }
            )
            logits, value, context = sess.run([self.on_logits, self.on_vf, self.on_lstm_state_out], feeder)
            logits = logits[0, ...]
            if self.ac_space.is_discrete:
                if deterministic:
                    sample = softmax(logits)

                else:
                    # Use multinomial to get sample (discrete):
                    sample = np.random.multinomial(1, softmax(logits))

                # print('ploicy_determ: {}, logits: {}, sample: {}'.format(deterministic, logits, sample))

                sample = self.ac_space._cat_to_vec(np.argmax(sample))

                # print('policy_sample_vector: ', sample)

            else:
                # Use DP to get sample (continuous):
                sample = sample_dp(logits, alpha=self.action_dp_alpha)

            # Get all needed action encodings:
            action = self.ac_space._vec_to_action(sample)
            one_hot = self.ac_space._vec_to_one_hot(sample)
            action_pack = {
                'environment': action,
                'encoded': self.ac_space.encode(action),
                'one_hot': one_hot,
            }
            # print('action_pack: ', action_pack)
        except Exception as e:
            print(e)
            raise e

        return action_pack, logits, value, context

    def get_value(self, observation, lstm_state, last_action, last_reward):
        """
        Estimates policy V-function.

        Args:
            observation:    single observation value
            lstm_state:     lstm context value
            last_action:    action value from previous step
            last_reward:    reward value from previous step

        Returns:
            V-function value
        """
        sess = tf.get_default_session()
        feeder = feed_dict_rnn_context(self.on_lstm_state_pl_flatten, lstm_state)
        feeder.update(feed_dict_from_nested(self.on_state_in, observation, expand_batch=True))
        feeder.update(
            {
                self.on_last_a_in: last_action,
                self.on_last_reward_in: last_reward,
                self.on_batch_size: 1,
                self.on_time_length: 1,
                self.train_phase: False
            }
        )

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

    @staticmethod
    def get_sample_config(*args, **kwargs):
        """
        Dummy implementation.

        Returns:
                default data sample configuration dictionary `btgym.datafeed.base.EnvResetConfig`
        """
        return EnvResetConfig


class Aac1dPolicy(BaseAacPolicy):
    """
    AAC policy for one-dimensional signal obs. state.
    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size,
                 lstm_class=rnn.BasicLSTMCell,
                 lstm_layers=(256,),
                 action_dp_alpha=200.0,
                 aux_estimate=True,
                 **kwargs):
        """
        Defines [partially shared] on/off-policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects bi-modal observation as dict: `external`, `internal`.

        Args:
            ob_space:           dictionary of observation state shapes
            ac_space:           discrete action space shape (length)
            rp_sequence_size:   reward prediction sample length
            lstm_class:         tf.nn.lstm class
            lstm_layers:        tuple of LSTM layers sizes
            aux_estimate:       (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary.
            **kwargs            not used
        """
        kwargs.update(
            dict(
                conv_2d_filter_size=[3, 1],
                conv_2d_stride=[2, 1],
                pc_estimator_stride=[2, 1],
                duell_pc_x_inner_shape=(6, 1, 32),  # [6,3,32] if swapping W-C dims
                duell_pc_filter_size=(4, 1),
                duell_pc_stride=(2, 1),
            )
        )
        super(Aac1dPolicy, self).__init__(
            ob_space,
            ac_space,
            rp_sequence_size,
            lstm_class,
            lstm_layers,
            action_dp_alpha,
            aux_estimate,
            **kwargs
        )

