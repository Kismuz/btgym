
from btgym.algorithms import BaseAAC
from btgym.algorithms.utils import feed_dict_rnn_context, feed_dict_from_nested, batch_stack
from .loss import guided_aac_loss_def_0_0, guided_aac_loss_def_0_1

import tensorflow as tf
import  numpy as np


class GA3C_0_0(BaseAAC):
    """
    Develop: Guided policy search framework to be.
    Stage 1.

    Tasks:
        Define and test local expert network and imitation loss on train data.
        Fit expert to train data locally. Perform several global parameter expert-aided updates.

    Want:
       convergence on train data.
    """
    def __init__(
            self,
            guided_loss_def=guided_aac_loss_def_0_1,
            aac_lambda=0.0,
            guided_lambda=1.0,
            guided_beta=0.01,
            cycles_per_trial=1,
            **kwargs
    ):
        """

        Args:
            guided_loss_def:                callable returning tensor holding imitation loss graph and summaries
            cycles_per_trial (int):         outer loop
            aac_lambda:
            guided_lambda:
            guided_beta:
            kwargs:                         BaseAAC kwargs
        """
        super(GA3C_0_0, self).__init__(_log_name='GuidedAAC_0.0', **kwargs)
        try:
            self.current_trial_num = -1
            self.cycles_per_trial = cycles_per_trial
            self.cycles_counter = 1

            # TODO: temporal plug here:
            self.guide = False

            with tf.device(self.worker_device):
                with tf.name_scope('local'):
                    # Make expert network:
                    self.expert_network = self._make_policy('local/expert')

                    # Op to copy weights from local policy to expert:
                    self.sync_expert = tf.group(
                        *[v1.assign(v2) for v1, v2 in zip(self.expert_network.var_list, self.local_network.var_list)]
                    )
                    # Override train_step op to be local network update:
                    self.train_op = self.optimizer.apply_gradients(
                        list(zip(self.grads, self.local_network.var_list))
                    )
                    # Make expert-assisted optimizer, loss and train op:
                    self.guided_optimizer = tf.train.AdamOptimizer(
                        self.learn_rate_decayed,
                        epsilon=1e-5,
                        name='local/guided/adam')

                    guided_loss, guided_summaries = guided_loss_def(
                        pi=self.local_network,
                        mu=self.expert_network,
                        entropy_beta=guided_beta,
                    )
                    # Compose new loss as sum of L_a3c + L_guided and define global parameters update op:
                    self.guided_loss = aac_lambda * self.loss + guided_lambda * guided_loss
                    self.guided_grads, _ = tf.clip_by_global_norm(
                        tf.gradients(self.guided_loss, self.local_network.var_list),
                        40.0,
                        name='local/guided/clip_grads'
                    )
                    self.guided_train_op = self.guided_optimizer.apply_gradients(
                        list(zip(self.guided_grads, self.network.var_list)),
                        name='local/guided/train_op'
                    )
                # Update model stat. summary:
                self.guided_model_summary_op = tf.summary.merge(guided_summaries, name='guided_model_summary')
        except:
            msg = 'Child class __init()__ exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def start(self, sess, summary_writer, **kwargs):
        """
        Executes all initializing operations,
        starts environment runner[s].
        Supposed to be called by parent worker just before training loop starts.

        Args:
            sess:           tf session object.
            kwargs:         not used by default.
        """
        try:
            # Copy weights: global -> local -> expert:
            sess.run(self.sync_pi)
            sess.run(self.sync_expert)
            # Start thread_runners:
            self._start_runners(sess, summary_writer)

        except:
            msg = 'Start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_sample_config(self):
        """
        Returns environment configuration parameters for next episode to sample.
        Controls Trials and Episodes data distributions.

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """
        #sess = tf.get_default_session()
        self.guide = False

        if self.current_train_episode < self.num_train_episodes:
            episode_type = 0  # train
            self.current_train_episode += 1
            self.log.info(
                'training, c_train={}, c_test={}, type={}'.
                format(self.current_train_episode, self.current_test_episode, episode_type)
            )
        else:
            if self.current_test_episode < self.num_test_episodes:
                # CHEAT: always train for v.0_0 but set `guided` flag:
                episode_type = 0  # test
                self.guide = True
                self.current_test_episode += 1
                self.log.info(
                    'guded-training, c_train={}, c_test={}, type={}'.
                    format(self.current_train_episode, self.current_test_episode, episode_type)
                )
            else:
                # single cycle end, reset counters:
                self.current_train_episode = 0
                self.current_test_episode = 0
                if self.cycles_counter < self.cycles_per_trial:
                    cycle_start_sample_config = dict(
                        episode_config=dict(
                            get_new=True,
                            sample_type=0,
                            b_alpha=1.0,
                            b_beta=1.0
                        ),
                        trial_config=dict(
                            get_new=False,
                            sample_type=0,
                            b_alpha=1.0,
                            b_beta=1.0
                        )
                    )
                    self.cycles_counter += 1
                    self.log.info(
                        'training (new cycle), c_train={}, c_test={}, type={}'.
                            format(self.current_train_episode, self.current_test_episode, 0)
                    )
                    return cycle_start_sample_config

                else:
                    init_sample_config = dict(
                        episode_config=dict(
                            get_new=True,
                            sample_type=0,
                            b_alpha=1.0,
                            b_beta=1.0
                        ),
                        trial_config=dict(
                            get_new=True,
                            sample_type=0,
                            b_alpha=1.0,
                            b_beta=1.0
                        )
                    )
                    self.cycles_counter = 1
                    self.log.info('new Trial at {}-th local iteration'.format(self.local_steps))
                    return init_sample_config

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=False,
                sample_type=0,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config

    def process_expert_data(self, data):
        """
        Processes data, composes guided train step sub-feed dictionary.
        Args:
            sess:               tf session obj.
            data (dict):        data dictionary

        Returns:
            feed_dict (dict):   expert policy network feed dictionary
        """
        on_policy_rollouts = data['on_policy']
        # TODO: redundant batch computation, mirrors base on-policy aac:
        on_policy_batch = batch_stack(
            [
                r.process(
                    gamma=self.model_gamma,
                    gae_lambda=self.model_gae_lambda,
                    size=self.rollout_length,
                    time_flat=self.time_flat,
                ) for r in on_policy_rollouts
            ]
        )
        # Sub-feeder for guided loss estimation graph:
        feed_dict = feed_dict_from_nested(self.expert_network.on_state_in, on_policy_batch['state'])
        feed_dict.update(
            feed_dict_rnn_context(self.expert_network.on_lstm_state_pl_flatten, on_policy_batch['context'])
        )
        feed_dict.update(
            {
                self.expert_network.on_a_r_in: on_policy_batch['last_action_reward'],
                self.expert_network.on_batch_size: on_policy_batch['batch_size'],
                self.expert_network.on_time_length: on_policy_batch['time_steps'],
                self.expert_network.train_phase: False,  # Zeroes learn rate, [+ batch_norm]
            }
        )

        return feed_dict

    def process(self, sess):
        """
        Cycle:
            1. Sync weights: local <- global
            2. Train expert to trial by [over]fitting local pi without global updates with A3C loss
            3. Update executive expert with fitted parameters: expert <- local
            4. Perform global training update: global <- Grad. guided loss
        """
        try:
            # Collect data from child thread runners:
            data = self._get_data()

            # Test or train: if at least one on-policy rollout from parallel runners is test one -
            # set entire minibatch to test. Doh.
            try:
                is_train = not np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any()

            except KeyError:
                is_train = True

            # New or same trial:
            # If at least one trial number from parallel runners has changed - assume new cycle start:
            # Pull trial number's from on_policy metadata:
            trial_num = np.asarray([env['state']['metadata']['trial_num'][-1] for env in data['on_policy']])
            if (trial_num != self.current_trial_num).any():
                # Copy global -> local -> expert:
                sess.run(self.sync_pi)
                sess.run(self.sync_expert)
                self.log.info(
                    'New Trial_{}, expert<-local<-global update at {}-th local iteration'.
                    format(trial_num, self.local_steps)
                )
                self.current_trial_num = trial_num

            feed_dict = self.process_data(sess, data, is_train=True)

            # Say `No` to redundant summaries:
            wirte_model_summary =\
                self.local_steps % self.model_summary_freq == 0

            if is_train and not self.guide:
                # Train expert locally:
                fetches = [self.train_op]

                self.log.info(
                    'local<-d.local update at {}-th local iteration'.format(self.local_steps)
                )
            else:
                # `Test` here means guided train step:
                # update global parameters with guided loss grads:
                sess.run(self.sync_pi)
                fetches = [self.guided_train_op, self.guided_model_summary_op]
                feed_dict.update(self.process_expert_data(data))

                self.log.info(
                    'GLOBAL<-d.guided_loss update at {}-th local iteration'.format(self.local_steps)
                )

            if wirte_model_summary:
                fetches_last = fetches + [self.model_summary_op, self.inc_step]
            else:
                fetches_last = fetches + [self.inc_step]

            # Do a number of SGD train epochs: HERE==1 !
            # When doing more than one epoch, we actually use only last summary:
            for i in range(self.num_epochs - 1):
                fetched = sess.run(fetches, feed_dict=feed_dict)

            fetched = sess.run(fetches_last, feed_dict=feed_dict)

            if is_train and not self.guide:
                # Back up local -> expert:
                sess.run(self.sync_expert)
                self.log.info(
                    'expert<-local update at {}-th local iteration'.format(self.local_steps)
                )
            else:
                # It has been guided train step:
                guided_model_summary = fetched[1]
                self.summary_writer.add_summary(tf.Summary.FromString(guided_model_summary), sess.run(self.global_step))
                self.summary_writer.flush()

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            # Write down summaries:
            self.process_summary(sess, data, model_summary)

            self.local_steps += 1

        except:
            msg = 'Train step exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)
