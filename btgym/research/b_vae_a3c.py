from tensorflow.contrib.layers import flatten as batch_flatten

from btgym.algorithms.policy.base import BaseAacPolicy
from btgym.algorithms.policy.stacked_lstm import AacStackedRL2Policy
from btgym.algorithms.nn.networks import *
from btgym.algorithms.utils import *
from btgym.algorithms.nn.layers import noisy_linear
from btgym.algorithms.nn.ae import beta_var_conv2d_autoencoder, conv2d_autoencoder

from btgym.algorithms import BaseAAC
from btgym.algorithms.nn.losses import beta_vae_loss_def, ae_loss_def


class bVAENPolicy(AacStackedRL2Policy):
    """
    Stacked LSTM with auxillary b-Variational AutoEncoder loss support and Noisy-net linear layers policy,
    based on `NAV A3C agent` architecture from

    `LEARNING TO NAVIGATE IN COMPLEX ENVIRONMENTS` by Mirowski et all. and

    `LEARNING TO REINFORCEMENT LEARN` by JX Wang et all.

    This policy class in conjunction with DataDomain classes from btgym.datafeed
    mimics RL^2 algorithm from

    `FAST REINFORCEMENT LEARNING VIA SLOW REINFORCEMENT LEARNING` by Duan et al.

    The difference is `get_initial_features()` method, which has been changed
    either to reset RNN context to zero-state or return context from the end of previous episode,
    depending on episode metadata received and `lstm_2_init_period' parameter.

    Papers:

    https://arxiv.org/pdf/1611.03673.pdf

    https://arxiv.org/pdf/1611.05763.pdf

    https://arxiv.org/abs/1706.10295.pdf

    https://arxiv.org/pdf/1611.02779.pdf

    """

    def __init__(self,
                 ob_space,
                 ac_space,
                 rp_sequence_size,
                 conv_2d_layer_config=(
                     (32, (3, 1), (2, 1)),
                     (32, (3, 1), (2, 1)),
                     (32, (3, 1), (2, 1)),
                     (32, (3, 1), (2, 1))
                 ),
                 lstm_class_ref=tf.contrib.rnn.LayerNormBasicLSTMCell,
                 lstm_layers=(256, 256),
                 lstm_2_init_period=50,
                 linear_layer_ref=noisy_linear,
                 encoder_class_ref=beta_var_conv2d_autoencoder,
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
            lstm_2_init_period: number of `get_initial_context()` method calls before force LSTM_2 context reset.
            linear_layer_ref:   linear layer class to use
            aux_estimate:       (bool), if True - add auxiliary tasks estimations to self.callbacks dictionary.
            **kwargs            not used
        """
        # 1D parameters override:
        # TODO: move to init kwargs
        kwargs.update(
            dict(
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

        # RL^2 related:
        self.current_trial_num = -1  # always give initial context at first call
        self.lstm_2_init_period = lstm_2_init_period
        self.current_ep_num = 0

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

        # ============= Base on-policy AAC network ===========

        # Conv. autoencoder:
        _, on_aac_x_ext, on_decoded_layers_ext, self.on_state_decoded_ext, on_d_kl_ext = encoder_class_ref(
            inputs=self.on_state_in['external'],
            layer_config=conv_2d_layer_config,
            linear_layer_ref=linear_layer_ref,
            max_batch_size=64,
            name='encoder_external',
            reuse=False
        )

        # VAE KL-divergence output:
        self.on_vae_d_kl_ext = on_d_kl_ext

        # Reshape rnn inputs for  batch training as [rnn_batch_dim, rnn_time_dim, flattened_depth]:
        x_shape_dynamic = tf.shape(on_aac_x_ext)
        max_seq_len = tf.cast(x_shape_dynamic[0] / self.on_batch_size, tf.int32)
        x_shape_static = on_aac_x_ext.get_shape().as_list()

        on_a_r_in = tf.reshape(self.on_a_r_in, [self.on_batch_size, max_seq_len, ac_space + 1])
        on_aac_x_ext = tf.reshape( on_aac_x_ext, [self.on_batch_size, max_seq_len, np.prod(x_shape_static[1:])])

        # Prepare `internal` state, if any:
        if 'internal' in list(self.on_state_in.keys()):
            if self.encode_internal_state:
                # Use convolution encoder:
                _, on_x_int, on_decoded_layers_int, self.on_state_decoded_int, on_d_kl_int = encoder_class_ref(
                    inputs=self.on_state_in['internal'],
                    layer_config=conv_2d_layer_config,
                    linear_layer_ref=linear_layer_ref,
                    max_batch_size=64,
                    name='encoder_internal',
                    reuse=False
                )
                # VAE KL-divergence output:
                self.on_vae_d_kl_int = on_d_kl_int

                x_int_shape_static = on_x_int.get_shape().as_list()
                on_x_int = [
                    tf.reshape(on_x_int, [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])]
                self.debug['state_internal_enc'] = tf.shape(on_x_int)

            else:
                # Feed as is:
                x_int_shape_static = self.on_state_in['internal'].get_shape().as_list()
                on_x_int = tf.reshape(
                    self.on_state_in['internal'],
                    [self.on_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                self.debug['state_internal'] = tf.shape(self.on_state_in['internal'])
                on_x_int = [on_x_int]
                self.on_state_decoded_int = None
                self.on_vae_d_kl_int = None

        else:
            on_x_int = []
            self.on_state_decoded_int = None
            self.on_vae_d_kl_int = None

        self.debug['conv_input_to_lstm1'] = tf.shape(on_aac_x_ext)

        # Feed last last_reward into LSTM_1 layer along with encoded `external` state features:
        on_stage2_1_input = [on_aac_x_ext, on_a_r_in[..., -1][..., None]] #+ on_x_internal

        # Feed last_action, encoded `external` state,  `internal` state into LSTM_2:
        on_stage2_2_input = [on_aac_x_ext, on_a_r_in] + on_x_int

        # LSTM_1 full input:
        on_aac_x_ext = tf.concat(on_stage2_1_input, axis=-1)

        self.debug['concat_input_to_lstm1'] = tf.shape(on_aac_x_ext)

        # First LSTM layer takes encoded `external` state:
        [on_x_lstm_1_out, self.on_lstm_1_init_state, self.on_lstm_1_state_out, self.on_lstm_1_state_pl_flatten] =\
            lstm_network(on_aac_x_ext, self.on_time_length, lstm_class_ref, (lstm_layers[0],), name='lstm_1')

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
        on_aac_x_ext = tf.concat(on_stage2_2_input, axis=-1)

        self.debug['on_stage2_2_input'] = tf.shape(on_aac_x_ext)

        [on_x_lstm_2_out, self.on_lstm_2_init_state, self.on_lstm_2_state_out, self.on_lstm_2_state_pl_flatten] = \
            lstm_network(on_aac_x_ext, self.on_time_length, lstm_class_ref, (lstm_layers[-1],), name='lstm_2')

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

        # ========= Off-policy AAC network (shared) ==========

        # Conv. autoencoder:
        _, off_aac_x, off_decoded_layers_ext, self.off_state_decoded_ext, off_d_kl_ext = encoder_class_ref(
            inputs=self.off_state_in['external'],
            layer_config=conv_2d_layer_config,
            linear_layer_ref=linear_layer_ref,
            max_batch_size=64,
            name='encoder_external',
            reuse=True
        )
        # VAE KL-divergence output:
        self.off_vae_d_kl_ext = off_d_kl_ext

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
                _, off_x_int, off_decoded_layers_int, self.off_state_decoded_int, off_d_kl_int = encoder_class_ref(
                    inputs=self.off_state_in['internal'],
                    layer_config=conv_2d_layer_config,
                    linear_layer_ref=linear_layer_ref,
                    max_batch_size=64,
                    name='encoder_internal',
                    reuse=True
                )
                self.off_vae_d_kl_int = off_d_kl_int

                x_int_shape_static = off_x_int.get_shape().as_list()
                off_x_int = [
                    tf.reshape(off_x_int, [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])])
                ]
            else:
                x_int_shape_static = self.off_state_in['internal'].get_shape().as_list()
                off_x_int = tf.reshape(
                    self.off_state_in['internal'],
                    [self.off_batch_size, max_seq_len, np.prod(x_int_shape_static[1:])]
                )
                off_x_int = [off_x_int]
                self.off_state_decoded_int = None
                self.off_vae_d_kl_int = None

        else:
            off_x_int = []
            self.off_state_decoded_int = None
            self.off_vae_d_kl_int = None

        off_stage2_1_input = [off_aac_x, off_a_r_in[..., -1][..., None]] #+ off_x_internal

        off_stage2_2_input = [off_aac_x, off_a_r_in] + off_x_int

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
        rp_encoded_layers, rp_x, rp_decoded_layers, _, rp_d_kl = encoder_class_ref(
            self.rp_state_in['external'],
            layer_config=conv_2d_layer_config,
            linear_layer_ref=linear_layer_ref,
            max_batch_size=64,
            name='encoder_external',
            reuse=True
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

                # UPD: keep both contexts:
                #new_context = context
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


class bVAENA3C(BaseAAC):

    def __init__(self, ae_loss=beta_vae_loss_def, ae_alpha=1.0, ae_beta=1.0, _log_name='bVAEN_A3C', **kwargs):
        try:
            super(bVAENA3C, self).__init__(name=_log_name, **kwargs)
            with tf.device(self.worker_device):
                with tf.variable_scope('local'):
                    on_vae_loss_ext, on_ae_summary_ext = ae_loss(
                        targets=self.local_network.on_state_in['external'],
                        logits=self.local_network.on_state_decoded_ext,
                        d_kl=self.local_network.on_vae_d_kl_ext,
                        alpha=ae_alpha,
                        beta=ae_beta,
                        name='external_state',
                        verbose=True
                    )
                    self.loss = self.loss + on_vae_loss_ext
                    extended_summary =[on_ae_summary_ext]

                    if self.local_network.encode_internal_state:
                        on_vae_loss_int, on_ae_summary_int = ae_loss(
                            targets=self.local_network.on_state_in['internal'],
                            logits=self.local_network.on_state_decoded_int,
                            d_kl=self.local_network.on_vae_d_kl_int,
                            alpha=ae_alpha,
                            beta=ae_beta,
                            name='internal_state',
                            verbose=True
                        )
                        self.loss = self.loss + on_vae_loss_int
                        extended_summary.append(on_ae_summary_int)

                    # Override train op def:
                    self.grads, _ = tf.clip_by_global_norm(
                        tf.gradients(self.loss, self.local_network.var_list),
                        40.0
                    )
                    grads_and_vars = list(zip(self.grads, self.network.var_list))
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)

                    # Merge summary:
                    extended_summary.append(self.model_summary_op)
                    self.model_summary_op = tf.summary.merge(extended_summary, name='extended_summary')

        except:
            msg = 'Child 0.0 class __init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)
