from tensorflow.contrib.layers import flatten as batch_flatten

from btgym.algorithms.policy.base import BaseAacPolicy
from btgym.algorithms.nn.networks import *
from btgym.algorithms.utils import *


class StackedLstmPolicy(BaseAacPolicy):
    """
    Conv.-Stacked_LSTM policy, based on `NAV A3C agent` architecture from

    `LEARNING TO NAVIGATE IN COMPLEX ENVIRONMENTS` by Mirowski et all. and

    `LEARNING TO REINFORCEMENT LEARN` by JX Wang et all.

    Papers:

    https://arxiv.org/pdf/1611.03673.pdf

    https://arxiv.org/pdf/1611.05763.pdf
    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size,
                 lstm_class_ref=tf.contrib.rnn.LayerNormBasicLSTMCell,
                 #lstm_class_ref=rnn.BasicLSTMCell,
                 lstm_layers=(256, 256),
                 linear_layer_ref=linear,
                 aux_estimate=False,
                 encode_internal_state=False,
                 **kwargs):
        """
        Defines [partially shared] on/off-policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects multi-modal observation as array of shape `ob_space`.

        Args:
            ob_space:           dictionary of observation state shapes
            ac_space:           discrete action space shape (length)
            rp_sequence_size:   reward prediction sample length
            lstm_class_ref:     tf.nn.lstm class to use
            lstm_layers:        tuple of LSTM layers sizes
            linear_layer_ref:   linear layer class to use
            aux_estimate:       (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary.
            **kwargs            not used
        """
        # 1D plug-in:
        kwargs.update(
            dict(
                conv_2d_filter_size=[3, 1],
                conv_2d_stride=[2, 1],
                conv_2d_num_filters=[32, 32, 64, 64],
                pc_estimator_stride=[2, 1],
                duell_pc_x_inner_shape=(6, 1, 32),  # [6,3,32] if swapping W-C dims
                duell_pc_filter_size=(4, 1),
                duell_pc_stride=(2, 1),
            )
        )

        self.ob_space = ob_space
        self.ac_space = ac_space
        self.rp_sequence_size = rp_sequence_size
        self.lstm_class = lstm_class_ref
        self.lstm_layers = lstm_layers
        self.aux_estimate = aux_estimate
        self.callback = {}
        self.encode_internal_state = encode_internal_state
        self.debug = {}

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
        on_aac_x = conv_2d_network(
            self.on_state_in['external'],
            ob_space['external'],
            ac_space,
            name='conv1d_external',
            **kwargs
        )

        # Aux min/max_loss:
        if 'raw_state' in list(self.on_state_in.keys()):
            self.raw_state = self.on_state_in['raw_state']
            self.state_min_max = tf.nn.elu(
                linear(
                    batch_flatten(on_aac_x),
                    2,
                    "min_max",
                    normalized_columns_initializer(0.01)
                )
            )
        else:
            self.raw_state = None
            self.state_min_max = None

            # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(on_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
        x_shape_static = on_aac_x.get_shape().as_list()

        on_a_r_in = tf.reshape(self.on_a_r_in, [self.on_batch_size, max_seq_len, ac_space + 1])
        on_aac_x = tf.reshape( on_aac_x, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # Prepare `internal` state, if any:
        if 'internal' in list(self.on_state_in.keys()):
            if self.encode_internal_state:
                # Use convolution encoder:
                on_x_internal = conv_2d_network(
                    self.on_state_in['internal'],
                    ob_space['internal'],
                    ac_space,
                    name='conv1d_internal',
                    # conv_2d_layer_ref=conv2d_dw,
                    conv_2d_num_filters=32,
                    conv_2d_num_layers=2,
                    conv_2d_filter_size=[3, 1],
                    conv_2d_stride=[2, 1],
                )
                x_int_shape_static = on_x_internal.get_shape().as_list()
                on_x_internal = [
                    tf.reshape(on_x_internal, [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])]
                self.debug['state_internal_enc'] = tf.shape(on_x_internal)

            else:
                # Feed as is:
                x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
                on_x_internal = tf.reshape(
                    self.on_state_in['internal'],
                    [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                self.debug['state_internal'] = tf.shape(self.on_state_in['internal'])
                on_x_internal = [on_x_internal]

        else:
            on_x_internal = []

        # Not used:
        if 'reward' in list(self.on_state_in.keys()):
            x_rewards_shape_static = self.on_state_in['reward'].get_shape().as_list()
            x_rewards = tf.reshape(
                self.on_state_in['reward'],
                [self.on_batch_size, max_seq_len, np.prod(x_rewards_shape_static[1:])]
            )
            self.debug['rewards'] = tf.shape(x_rewards)
            x_rewards = [x_rewards]

        else:
            x_rewards = []

        self.debug['conv_input_to_lstm1'] = tf.shape(on_aac_x)

        # Feed last last_reward into LSTM_1 layer along with encoded `external` state features:
        on_stage2_1_input = [on_aac_x, on_a_r_in[..., -1][..., None]] #+ on_x_internal

        # Feed last_action, encoded `external` state,  `internal` state into LSTM_2:
        on_stage2_2_input = [on_aac_x, on_a_r_in] + on_x_internal

        # LSTM_1 full input:
        on_aac_x = tf.concat(on_stage2_1_input, axis=-1)

        self.debug['concat_input_to_lstm1'] = tf.shape(on_aac_x)

        # First LSTM layer takes encoded `external` state:
        [on_x_lstm_1_out, self.on_lstm_1_init_state, self.on_lstm_1_state_out, self.on_lstm_1_state_pl_flatten] =\
            lstm_network(on_aac_x, self.on_time_length, lstm_class_ref, (lstm_layers[0],), name='lstm_1')

        self.debug['on_x_lstm_1_out'] = tf.shape(on_x_lstm_1_out)
        self.debug['self.on_lstm_1_state_out'] = tf.shape(self.on_lstm_1_state_out)
        self.debug['self.on_lstm_1_state_pl_flatten'] = tf.shape(self.on_lstm_1_state_pl_flatten)

        # For time_flat only: Reshape on_lstm_1_state_out from [1,2,20,size] -->[20,1,2,size] --> [20,1, 2xsize]:
        reshape_lstm_1_state_out = tf.transpose(self.on_lstm_1_state_out, [2, 0, 1, 3])
        reshape_lstm_1_state_out_shape_static = reshape_lstm_1_state_out.get_shape().as_list()
        reshape_lstm_1_state_out = tf.reshape(
            reshape_lstm_1_state_out,
            [self.on_batch_size, max_seq_len, np.prod(reshape_lstm_1_state_out_shape_static[-2:])],
        )
        #self.debug['reshape_lstm_1_state_out'] = tf.shape(reshape_lstm_1_state_out)

        # Take policy logits off first LSTM-dense layer:
        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_1_out.get_shape().as_list()
        rsh_on_x_lstm_1_out = tf.reshape(on_x_lstm_1_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_1_out'] = tf.shape(rsh_on_x_lstm_1_out)

        # Aac policy output and action-sampling function:
        [self.on_logits, _, self.on_sample] = dense_aac_network(
            rsh_on_x_lstm_1_out,
            ac_space,
            linear_layer_ref=linear_layer_ref,
            name='aac_dense_pi'
        )

        # Second LSTM layer takes concatenated encoded 'external' state, LSTM_1 output,
        # last_action and `internal_state` (if present) tensors:
        on_stage2_2_input += [on_x_lstm_1_out]

        # Try: feed context instead of output
        #on_stage2_2_input = [reshape_lstm_1_state_out] + on_stage2_1_input

        # LSTM_2 full input:
        on_aac_x = tf.concat(on_stage2_2_input, axis=-1)

        self.debug['on_stage2_2_input'] = tf.shape(on_aac_x)

        [on_x_lstm_2_out, self.on_lstm_2_init_state, self.on_lstm_2_state_out, self.on_lstm_2_state_pl_flatten] = \
            lstm_network(on_aac_x, self.on_time_length, lstm_class_ref, (lstm_layers[-1],), name='lstm_2')

        self.debug['on_x_lstm_2_out'] = tf.shape(on_x_lstm_2_out)
        self.debug['self.on_lstm_2_state_out'] = tf.shape(self.on_lstm_2_state_out)
        self.debug['self.on_lstm_2_state_pl_flatten'] = tf.shape(self.on_lstm_2_state_pl_flatten)

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_2_out.get_shape().as_list()
        on_x_lstm_out = tf.reshape(on_x_lstm_2_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_out'] = tf.shape(on_x_lstm_out)

        # Aac value function:
        [_, self.on_vf, _] = dense_aac_network(
            on_x_lstm_out,
            ac_space,
            linear_layer_ref=linear_layer_ref,
            name='aac_dense_vfn'
        )

        # Concatenate LSTM placeholders, init. states and context:
        self.on_lstm_init_state = (self.on_lstm_1_init_state, self.on_lstm_2_init_state)
        self.on_lstm_state_out = (self.on_lstm_1_state_out, self.on_lstm_2_state_out)
        self.on_lstm_state_pl_flatten = self.on_lstm_1_state_pl_flatten + self.on_lstm_2_state_pl_flatten


        #if False: # Temp. disable

        # Off-policy AAC network (shared):
        off_aac_x = conv_2d_network(
            self.off_state_in['external'],
            ob_space['external'],
            ac_space,
            name='conv1d_external',
            reuse=True,
            **kwargs
        )
        # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(off_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.off_batch_size, tf.int32)
        x_shape_static = off_aac_x.get_shape().as_list()

        off_a_r_in = tf.reshape(self.off_a_r_in, [self.off_batch_size, max_seq_len, ac_space + 1])
        off_aac_x = tf.reshape( off_aac_x, [self.off_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # Prepare `internal` state, if any:
        if 'internal' in list(self.off_state_in.keys()):
            if self.encode_internal_state:
                # Use convolution encoder:
                off_x_internal = conv_2d_network(
                    self.off_state_in['internal'],
                    ob_space['internal'],
                    ac_space,
                    name='conv1d_internal',
                    # conv_2d_layer_ref=conv2d_dw,
                    conv_2d_num_filters=32,
                    conv_2d_num_layers=2,
                    conv_2d_filter_size=[3, 1],
                    conv_2d_stride=[2, 1],
                    reuse=True,
                )
                x_int_shape_static = off_x_internal.get_shape().as_list()
                off_x_internal = [
                    tf.reshape(off_x_internal, [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])
                ]
            else:
                x_int_shape_static = self.off_state_in['internal'].get_shape().as_list()
                off_x_internal = tf.reshape(
                    self.off_state_in['internal'],
                    [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                off_x_internal = [off_x_internal]

        else:
            off_x_internal = []

        off_stage2_1_input = [off_aac_x, off_a_r_in[..., -1][..., None]] #+ off_x_internal

        off_stage2_2_input = [off_aac_x, off_a_r_in] + off_x_internal

        off_aac_x = tf.concat(off_stage2_1_input, axis=-1)

        [off_x_lstm_1_out, _, _, self.off_lstm_1_state_pl_flatten] =\
            lstm_network(off_aac_x, self.off_time_length, lstm_class_ref, (lstm_layers[0],), name='lstm_1', reuse=True)

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = off_x_lstm_1_out.get_shape().as_list()
        rsh_off_x_lstm_1_out = tf.reshape(off_x_lstm_1_out, [x_shape_dynamic[0], x_shape_static[-1]])

        [self.off_logits, _, _] =\
            dense_aac_network(
                rsh_off_x_lstm_1_out,
                ac_space,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_pi',
                reuse=True
            )

        off_stage2_2_input += [off_x_lstm_1_out]

        # LSTM_2 full input:
        off_aac_x = tf.concat(off_stage2_2_input, axis=-1)

        [off_x_lstm_2_out, _, _, self.off_lstm_2_state_pl_flatten] = \
            lstm_network(off_aac_x, self.off_time_length, lstm_class_ref, (lstm_layers[-1],), name='lstm_2', reuse=True)

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = off_x_lstm_2_out.get_shape().as_list()
        off_x_lstm_out = tf.reshape(off_x_lstm_2_out, [x_shape_dynamic[0], x_shape_static[-1]])

        # Aac value function:
        [_, self.off_vf, _] = dense_aac_network(
            off_x_lstm_out,
            ac_space,
            linear_layer_ref=linear_layer_ref,
            name='aac_dense_vfn',
            reuse=True
        )

        # Concatenate LSTM states:
        self.off_lstm_state_pl_flatten = self.off_lstm_1_state_pl_flatten + self.off_lstm_2_state_pl_flatten

        # Aux1:
        # `Pixel control` network.
        #
        # Define pixels-change estimation function:
        # Yes, it rather env-specific but for atari case it is handy to do it here, see self.get_pc_target():
        [self.pc_change_state_in, self.pc_change_last_state_in, self.pc_target] =\
            pixel_change_2d_estimator(ob_space['external'], **kwargs)

        self.pc_batch_size = self.off_batch_size
        self.pc_time_length = self.off_time_length

        self.pc_state_in = self.off_state_in
        self.pc_a_r_in = self.off_a_r_in
        self.pc_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten

        # Shared conv and lstm nets, same off-policy batch:
        pc_x = off_x_lstm_out

        # PC duelling Q-network, outputs [None, 20, 20, ac_size] Q-features tensor:
        self.pc_q = duelling_pc_network(pc_x, self.ac_space, linear_layer_ref=linear_layer_ref, **kwargs)

        # Aux2:
        # `Value function replay` network.
        #
        # VR network is fully shared with ppo network but with `value` only output:
        # and has same off-policy batch pass with off_ppo network:
        self.vr_batch_size = self.off_batch_size
        self.vr_time_length = self.off_time_length

        self.vr_state_in = self.off_state_in
        self.vr_a_r_in = self.off_a_r_in

        self.vr_lstm_state_pl_flatten = self.off_lstm_state_pl_flatten
        self.vr_value = self.off_vf

        # Aux3:
        # `Reward prediction` network.
        self.rp_batch_size = tf.placeholder(tf.int32, name='rp_batch_size')

        # Shared conv. output:
        rp_x = conv_2d_network(
            self.rp_state_in['external'],
            ob_space['external'],
            ac_space,
            name='conv1d_external',
            reuse=True,
            **kwargs
        )

        # Flatten batch-wise:
        rp_x_shape_static = rp_x.get_shape().as_list()
        rp_x = tf.reshape(rp_x, [self.rp_batch_size, np.prod(rp_x_shape_static[1:]) * (self.rp_sequence_size-1)])

        # RP output:
        self.rp_logits = dense_rp_network(rp_x, linear_layer_ref=linear_layer_ref)

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

        # Callbacks:
        if self.aux_estimate:
            self.callback['pixel_change'] = self.get_pc_target


class AacStackedRL2Policy(StackedLstmPolicy):
    """
    Attempt to implement two-level RL^2
    This policy class in conjunction with DataDomain classes from btgym.datafeed
    is aimed to implement RL^2 algorithm by Duan et al.

    Paper:
    `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING`,
    https://arxiv.org/pdf/1611.02779.pdf

    The only difference from Base policy is `get_initial_features()` method, which has been changed
    either to reset RNN context to zero-state or return context from the end of previous episode,
    depending on episode metadata received or `lstm_2_init_period' parameter.
    """
    def __init__(self, lstm_2_init_period=50, **kwargs):
        super(AacStackedRL2Policy, self).__init__(**kwargs)
        self.current_trial_num = -1  # always give initial context at first call
        self.lstm_2_init_period = lstm_2_init_period
        self.current_ep_num = 0

    def get_initial_features(self, state, context=None):
        """
        Returns RNN initial context.
        RNN_1 (lower) context is reset at every call.

        RNN_2 (upper) context is reset:
            - every `lstm_2_init_period' episodes;
            - episode  initial `state` `trial_num` metadata has been changed form last call (new train trial started);
            - episode metatdata `type` is non-zero (test episode);
            - no context arg is provided (initial episode of training);
            - ... else carries context on to new episode;

        Episode metadata are provided by DataTrialIterator, which is shaping Trial data distribution in this case,
        and delivered through env.strategy as separate key in observation dictionary.

        Args:
            state:      initial episode state (result of env.reset())
            context:    last previous episode RNN state (last_context of runner)

        Returns:
            2_RNN zero-state tuple.

        Raises:
            KeyError if [`metadata`]:[`trial_num`,`type`] keys not found
        """
        try:
            sess = tf.get_default_session()
            new_context = list(sess.run(self.on_lstm_init_state))
            if state['metadata']['trial_num'] != self.current_trial_num\
                    or context is None\
                    or state['metadata']['type']\
                    or self.current_ep_num % self.lstm_2_init_period == 0:
                # Assume new/initial trial or test sample, reset_1, 2 context:
                pass #print('RL^2 policy context 1, 2 reset')

            else:
                # Asssume same training trial, keep context_2 same as received:
                new_context[-1] = context[-1]
                #print('RL^2 policy context 1, reset')
            # Back to tuple:
            new_context = tuple(new_context)
            # Keep trial number:
            self.current_trial_num = state['metadata']['trial_num']

        except KeyError:
            raise KeyError(
                'RL^2 policy: expected observation state dict. to have keys [`metadata`]:[`trial_num`,`type`]; got: {}'.
                format(state.keys())
            )
        self.current_ep_num +=1
        return new_context

