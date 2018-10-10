from btgym.algorithms.policy.base import BaseAacPolicy
from btgym.algorithms.nn.networks import *
from btgym.algorithms.utils import *
from btgym.spaces import DictSpace, ActionDictSpace

from btgym.algorithms.math_utils import sample_dp, softmax


class RegressionTestPolicy(BaseAacPolicy):
    """
    Simplified LSTM policy (off-policy training excluded, regression heads added)
    TODO: remove last_action, last_reward terms
    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size=4,
                 state_encoder_class_ref=conv_2d_network,
                 lstm_class_ref=tf.contrib.rnn.LayerNormBasicLSTMCell,
                 lstm_layers=(256, 256),
                 linear_layer_ref=noisy_linear,
                 share_encoder_params=False,
                 dropout_keep_prob=1.0,
                 action_dp_alpha=200.0,
                 aux_estimate=False,
                 encode_internal_state=False,
                 static_rnn=False,
                 shared_p_v=False,
                 lstm_2_init_period=50,
                 regression_type='simple',
                 **kwargs):
        """
        Defines [partially shared] on/policy networks for estimating  action-logits, value function,
        reward and state 'pixel_change' predictions.
        Expects multi-modal observation as array of shape `ob_space`.

        Args:
            ob_space:               instance of btgym.spaces.DictSpace
            ac_space:               instance of btgym.spaces.ActionDictSpace
            rp_sequence_size:       reward prediction sample length
            lstm_class_ref:         tf.nn.lstm class to use
            lstm_layers:            tuple of LSTM layers sizes
            linear_layer_ref:       linear layer class to use
            share_encoder_params:   bool, whether to share encoder parameters for every 'external' data stream
            dropout_keep_prob:      in (0, 1] dropout regularisation parameter
            action_dp_alpha:
            aux_estimate:           (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary
            encode_internal_state:  use encoder over 'internal' part of observation space
            static_rnn:             (bool), it True - use static rnn graph, dynamic otherwise
            lstm_2_init_period:     int, RL2 single trial size
            regression_type:        str, regression type (currently: 'simple', 'rnn')
            **kwargs                not used
        """

        assert isinstance(ob_space, DictSpace), \
            'Expected observation space be instance of btgym.spaces.DictSpace, got: {}'.format(ob_space)
        self.ob_space = ob_space

        assert isinstance(ac_space, ActionDictSpace), \
            'Expected action space be instance of btgym.spaces.ActionDictSpace, got: {}'.format(ac_space)

        self.ac_space = ac_space

        # self.rp_sequence_size = rp_sequence_size
        self.state_encoder_class_ref = state_encoder_class_ref
        self.lstm_class = lstm_class_ref
        self.lstm_layers = lstm_layers
        self.action_dp_alpha = action_dp_alpha
        self.aux_estimate = aux_estimate
        self.callback = {}
        self.encode_internal_state = encode_internal_state
        self.share_encoder_params = share_encoder_params
        if self.share_encoder_params:
            self.reuse_encoder_params = tf.AUTO_REUSE

        else:
            self.reuse_encoder_params = False
        self.static_rnn = static_rnn
        self.dropout_keep_prob = dropout_keep_prob
        assert 0 < self.dropout_keep_prob <= 1, 'Dropout keep_prob value should be in (0, 1]'

        self.regression_type = regression_type

        self.debug = {}

        # Placeholders for obs. state input:
        self.on_state_in = nested_placeholders(self.ob_space.shape, batch_dim=None, name='on_policy_state_in')

        # Placeholders for previous step action[multi-categorical vector encoding]  and reward [scalar]:
        self.on_last_a_in = tf.placeholder(
            tf.float32,
            [None, self.ac_space.encoded_depth],
            name='on_policy_last_action_in_pl'
        )
        self.on_last_reward_in = tf.placeholder(tf.float32, [None], name='on_policy_last_reward_in_pl')

        # Placeholders for rnn batch and time-step dimensions:
        self.on_batch_size = tf.placeholder(tf.int32, name='on_policy_batch_size')
        self.on_time_length = tf.placeholder(tf.int32, name='on_policy_sequence_size')

        self.debug['on_state_in_keys'] = list(self.on_state_in.keys())

        assert 'regression_targets' in self.on_state_in.keys(), 'Obs. space should provide regression targets.'
        self.regression_targets = self.on_state_in['regression_targets']
        self.debug['regression_targets'] = self.regression_targets

        # Dropout related:
        try:
            if self.train_phase is not None:
                pass

        except AttributeError:
            self.train_phase = tf.placeholder_with_default(
                tf.constant(False, dtype=tf.bool),
                shape=(),
                name='train_phase_flag_pl'
            )
        self.keep_prob = 1.0 - (1.0 - self.dropout_keep_prob) * tf.cast(self.train_phase, tf.float32)

        # Default parameters:
        default_kwargs = dict(
            conv_2d_filter_size=[3, 1],
            conv_2d_stride=[2, 1],
            conv_2d_num_filters=[32, 32, 64, 64],
            pc_estimator_stride=[2, 1],
            duell_pc_x_inner_shape=(6, 1, 32),  # [6,3,32] if swapping W-C dims
            duell_pc_filter_size=(4, 1),
            duell_pc_stride=(2, 1),
            keep_prob=self.keep_prob,
        )
        # Insert if not already:
        for key, default_value in default_kwargs.items():
            if key not in kwargs.keys():
                kwargs[key] = default_value

        # Base on-policy AAC network:

        # Separately encode than concatenate all `external` states modes, jointly encode every stream within mode:
        self.on_aac_x_encoded = {}
        for key in self.on_state_in.keys():
            if 'external' in key:
                if isinstance(self.on_state_in[key], dict):  # got dictionary of data streams
                    if self.share_encoder_params:
                        layer_name_template = 'encoded_{}_shared'
                    else:
                        layer_name_template = 'encoded_{}_{}'
                    encoded_streams = {
                        name: tf.layers.flatten(
                            self.state_encoder_class_ref(
                                x=stream,
                                ob_space=self.ob_space.shape[key][name],
                                ac_space=self.ac_space,
                                name=layer_name_template.format(key, name),
                                reuse=self.reuse_encoder_params,  # shared params for all streams in mode
                                **kwargs
                            )
                        )
                        for name, stream in self.on_state_in[key].items()
                    }
                    encoded_mode = tf.concat(
                        list(encoded_streams.values()),
                        axis=-1,
                        name='multi_encoded_{}'.format(key)
                    )
                else:
                    # Got single data stream:
                    encoded_mode = tf.layers.flatten(
                        self.state_encoder_class_ref(
                            x=self.on_state_in[key],
                            ob_space=self.ob_space.shape[key],
                            ac_space=self.ac_space,
                            name='encoded_{}'.format(key),
                            **kwargs
                        )
                    )
                self.on_aac_x_encoded[key] = encoded_mode

        self.debug['on_state_external_encoded_dict'] = self.on_aac_x_encoded

        on_aac_x = tf.concat(list(self.on_aac_x_encoded.values()), axis=-1, name='on_state_external_encoded')

        self.debug['on_state_external_encoded'] = on_aac_x

        # TODO: for encoder prediction test, output `naive` estimates for logits and value directly from encoder:
        [self.on_simple_logits, self.on_simple_value, _] = dense_aac_network(
            tf.layers.flatten(on_aac_x),
            ac_space_depth=self.ac_space.one_hot_depth,
            linear_layer_ref=linear_layer_ref,
            name='aac_dense_simple_pi_v'
        )

        # Reshape rnn inputs for batch training as: [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(on_aac_x)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
        x_shape_static = on_aac_x.get_shape().as_list()

        on_last_action_in = tf.reshape(
            self.on_last_a_in,
            [self.on_batch_size, max_seq_len, self.ac_space.encoded_depth]
        )
        on_last_r_in = tf.reshape(self.on_last_reward_in, [self.on_batch_size, max_seq_len, 1])

        on_aac_x = tf.reshape(on_aac_x, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # Prepare `internal` state, if any:
        if 'internal' in list(self.on_state_in.keys()):
            if self.encode_internal_state:
                # Use convolution encoder:
                on_x_internal = self.state_encoder_class_ref(
                    x=self.on_state_in['internal'],
                    ob_space=self.ob_space.shape['internal'],
                    ac_space=self.ac_space,
                    name='encoded_internal',
                    **kwargs
                )
                x_int_shape_static = on_x_internal.get_shape().as_list()
                on_x_internal = [
                    tf.reshape(on_x_internal, [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])]
                self.debug['on_state_internal_encoded'] = on_x_internal

            else:
                # Feed as is:
                x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
                on_x_internal = tf.reshape(
                    self.on_state_in['internal'],
                    [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                self.debug['on_state_internal_encoded'] = on_x_internal
                on_x_internal = [on_x_internal]

        else:
            on_x_internal = []

        # Prepare datetime index if any:
        if 'datetime' in list(self.on_state_in.keys()):
            x_dt_shape_static = self.on_state_in['datetime'].get_shape().as_list()
            on_x_dt = tf.reshape(
                self.on_state_in['datetime'],
                [self.on_batch_size, max_seq_len, np.prod(x_dt_shape_static[1:])]
            )
            on_x_dt = [on_x_dt]

        else:
            on_x_dt = []

        self.debug['on_state_dt_encoded'] = on_x_dt
        self.debug['conv_input_to_lstm1'] = on_aac_x

        # Feed last last_reward into LSTM_1 layer along with encoded `external` state features and datetime stamp:
        # on_stage2_1_input = [on_aac_x, on_last_action_in, on_last_reward_in] + on_x_dt
        on_stage2_1_input = [on_aac_x, on_last_r_in] #+ on_x_dt

        # Feed last_action, encoded `external` state,  `internal` state, datetime stamp into LSTM_2:
        # on_stage2_2_input = [on_aac_x, on_last_action_in, on_last_reward_in] + on_x_internal + on_x_dt
        on_stage2_2_input = [on_aac_x, on_last_action_in] + on_x_internal #+ on_x_dt

        # LSTM_1 full input:
        on_aac_x = tf.concat(on_stage2_1_input, axis=-1)

        self.debug['concat_input_to_lstm1'] = on_aac_x

        # First LSTM layer takes encoded `external` state:
        [on_x_lstm_1_out, self.on_lstm_1_init_state, self.on_lstm_1_state_out, self.on_lstm_1_state_pl_flatten] =\
            lstm_network(
                x=on_aac_x,
                lstm_sequence_length=self.on_time_length,
                lstm_class=lstm_class_ref,
                lstm_layers=(lstm_layers[0],),
                static=static_rnn,
                dropout_keep_prob=self.dropout_keep_prob,
                name='lstm_1'
            )

        # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # print('var_list: ', var_list)

        self.debug['on_x_lstm_1_out'] = on_x_lstm_1_out
        self.debug['self.on_lstm_1_state_out'] = self.on_lstm_1_state_out
        self.debug['self.on_lstm_1_state_pl_flatten'] = self.on_lstm_1_state_pl_flatten

        # For time_flat only: Reshape on_lstm_1_state_out from [1,2,20,size] -->[20,1,2,size] --> [20,1, 2xsize]:
        reshape_lstm_1_state_out = tf.transpose(self.on_lstm_1_state_out, [2, 0, 1, 3])
        reshape_lstm_1_state_out_shape_static = reshape_lstm_1_state_out.get_shape().as_list()

        # Take policy logits off first LSTM-dense layer:
        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_1_out.get_shape().as_list()
        rsh_on_x_lstm_1_out = tf.reshape(on_x_lstm_1_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_1_out'] = rsh_on_x_lstm_1_out

        if not shared_p_v:
            # Aac policy output and action-sampling function:
            [self.on_logits, _, self.on_sample] = dense_aac_network(
                rsh_on_x_lstm_1_out,
                ac_space_depth=self.ac_space.one_hot_depth,
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

        self.debug['on_stage2_2_input'] = on_aac_x

        [on_x_lstm_2_out, self.on_lstm_2_init_state, self.on_lstm_2_state_out, self.on_lstm_2_state_pl_flatten] = \
            lstm_network(
                x=on_aac_x,
                lstm_sequence_length=self.on_time_length,
                lstm_class=lstm_class_ref,
                lstm_layers=(lstm_layers[-1],),
                static=static_rnn,
                dropout_keep_prob=self.dropout_keep_prob,
                name='lstm_2'
            )

        self.debug['on_x_lstm_2_out'] = on_x_lstm_2_out
        self.debug['self.on_lstm_2_state_out'] = self.on_lstm_2_state_out
        self.debug['self.on_lstm_2_state_pl_flatten'] = self.on_lstm_2_state_pl_flatten

        # Reshape back to [batch, flattened_depth], where batch = rnn_batch_dim * rnn_time_dim:
        x_shape_static = on_x_lstm_2_out.get_shape().as_list()
        rsh_on_x_lstm_2_out = tf.reshape(on_x_lstm_2_out, [x_shape_dynamic[0], x_shape_static[-1]])

        self.debug['reshaped_on_x_lstm_2_out'] = rsh_on_x_lstm_2_out

        if shared_p_v:
            # Take pi an value fn. estimates off second LSTM layer:
            [self.on_logits, self.on_vf, self.on_sample] = dense_aac_network(
                rsh_on_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_pi_vfn'
            )

        else:
            # Take pi off first LSTM layer, an value off second:
            [_, self.on_vf, _] = dense_aac_network(
                rsh_on_x_lstm_2_out,
                ac_space_depth=self.ac_space.one_hot_depth,
                linear_layer_ref=linear_layer_ref,
                name='aac_dense_vfn'
            )

        # Add test regression/ classification heads:
        if self.regression_type == 'simple':
            # Naive regression (off encoder output):
            # [self.regression, _, _] = dense_aac_network(
            #     # tf.layers.flatten(self.debug['conv_input_to_lstm1']),
            #     tf.layers.flatten(self.debug['on_state_external_encoded']),
            #     # ac_space_depth=self.regression_depth,
            #     ac_space_depth=self.regression_targets.shape.as_list()[-1],
            #     linear_layer_ref=linear_layer_ref,
            #     name='on_dense_simple_regression'
            # )
            #self.regression = tf.layers.flatten(self.debug['on_state_external_encoded'])
            self.regression = linear(
                x=tf.layers.flatten(self.debug['on_state_external_encoded']),
                size=self.regression_targets.shape.as_list()[-1],
                initializer=normalized_columns_initializer(0.1),
                name='on_dense_simple_regression',
                reuse=False
            )
        elif self.regression_type == 'rnn':
            # Context-aware regression (off LSTM bank):
            # [self.regression, _, _] = dense_aac_network(
            #     tf.layers.flatten(self.debug['reshaped_on_x_lstm_2_out']),
            #     # ac_space_depth=self.regression_depth,
            #     ac_space_depth=self.regression_targets.shape.as_list()[-1],
            #     linear_layer_ref=linear_layer_ref,
            #     name='on_dense_rnn_regression'
            # )
            self.regression = linear(
                x=tf.layers.flatten(self.debug['reshaped_on_x_lstm_2_out']),
                size=self.regression_targets.shape.as_list()[-1],
                initializer=normalized_columns_initializer(0.1),
                name='on_dense_rnn_regression',
                reuse=False
            )
        else:
            raise NotImplementedError('Unknown regression type `{}`'.format(self.regression_type))

        # Concatenate LSTM placeholders, init. states and context:
        self.on_lstm_init_state = (self.on_lstm_1_init_state, self.on_lstm_2_init_state)
        self.on_lstm_state_out = (self.on_lstm_1_state_out, self.on_lstm_2_state_out)
        self.on_lstm_state_pl_flatten = self.on_lstm_1_state_pl_flatten + self.on_lstm_2_state_pl_flatten

        # Batch-norm related:
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # Add moving averages to save list:
        moving_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*moving.*')
        renorm_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*renorm.*')

        # What to save:
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.var_list += moving_var_list + renorm_var_list

        # RL2 related:
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

    def act(self, observation, lstm_state, last_action, last_reward):
        """
        Predicts action.

        Args:
            observation:    dictionary containing single observation
            lstm_state:     lstm context value
            last_action:    action value from previous step
            last_reward:    reward value previous step

        Returns:
            Action as dictionary of several action encodings, actions logits, V-fn value, output RNN state
        """
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
        # action_one_hot, logits, value, context = sess.run(
        #     [self.on_sample, self.on_logits, self.on_vf, self.on_lstm_state_out],
        #     feeder
        # )
        # return action_one_hot, logits, value, context
        logits, value, context, regression = sess.run(
            [self.on_logits, self.on_vf, self.on_lstm_state_out, self.regression],
            feeder
        )
        logits = logits[0, ...]
        if self.ac_space.is_discrete:
            # Use multinomial to get sample (discrete):
            sample = np.random.multinomial(1, softmax(logits))
            sample = self.ac_space._cat_to_vec(np.argmax(sample))

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

        return action_pack, logits, value, context, regression

