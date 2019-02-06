import sys
from logbook import Logger, StreamHandler

import numpy as np
import tensorflow as tf
from btgym.research.gps.aac import GuidedAAC
from .runner import OUpRunner


class OUpAAC(GuidedAAC):
    """
    Extends parent `GuidedAAC` class with additional summaries related to Orn-Uhl. data generating process.
    """

    def __init__(
            self,
            runner_config=None,
            aac_lambda=1.0,
            guided_lambda=0.0,
            name='OUpA3C',
            **kwargs
    ):
        if runner_config is None:
            runner_config = {
                'class_ref': OUpRunner,
                'kwargs': {}
            }
        super(OUpAAC, self).__init__(
            aac_lambda=aac_lambda,
            guided_lambda=guided_lambda,
            name=name,
            runner_config=runner_config,
            **kwargs,
        )

    def _combine_summaries(self, policy=None, model_summaries=None):
        """
        Defines model-wide and episode-related summaries

        Returns:
            model_summary op
            episode_summary op
        """
        if model_summaries is not None:
            if self.use_global_network:
                # Model-wide statistics:
                with tf.name_scope('model'):
                    model_summaries += [
                        tf.summary.scalar("grad_global_norm", self.grads_global_norm),
                        tf.summary.scalar("learn_rate", self.learn_rate_decayed),
                        # cause actual rate is a jaggy due to test freezes
                        tf.summary.scalar("total_loss", self.loss),
                    ]
                    if policy is not None:
                        model_summaries += [tf.summary.scalar("var_global_norm", tf.global_norm(policy.var_list))]
        else:
            model_summaries = []
        # Model stat. summary:
        model_summary = tf.summary.merge(model_summaries, name='model_summary')

        # Episode-related summaries:
        ep_summary = dict(
            # Summary placeholders
            render_atari=tf.placeholder(tf.uint8, [None, None, None, 1]),
            total_r=tf.placeholder(tf.float32, ),
            cpu_time=tf.placeholder(tf.float32, ),
            final_value=tf.placeholder(tf.float32, ),
            steps=tf.placeholder(tf.int32, ),
            ou_lambda=tf.placeholder(tf.float32, ),
            ou_sigma=tf.placeholder(tf.float32, ),
            ou_mu=tf.placeholder(tf.float32, ),

        )
        ep_summary.update(
            {
                mode: tf.placeholder(tf.uint8, [None, None, None, None], name=mode + '_pl')
                for mode in self.env_list[0].render_modes + self.aux_render_modes
            }
        )
        ep_summary['render_op'] = tf.summary.merge(
            [tf.summary.image(mode, ep_summary[mode])
             for mode in self.env_list[0].render_modes + self.aux_render_modes]
        )
        # Episode stat. summary:
        ep_summary['btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_train/cpu_time_sec', ep_summary['cpu_time']),
                tf.summary.scalar('episode_train/final_value', ep_summary['final_value']),
                tf.summary.scalar('episode_train/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_train/ou_lambda', ep_summary['ou_lambda']),
                tf.summary.scalar('episode_train/ou_sigma', ep_summary['ou_sigma']),
                tf.summary.scalar('episode_train/ou_mu', ep_summary['ou_mu']),
            ],
            name='episode_train_btgym'
        )
        # Test episode stat. summary:
        ep_summary['test_btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_test/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_test/final_value', ep_summary['final_value']),
            ],
            name='episode_test_btgym'
        )
        ep_summary['atari_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode/steps', ep_summary['steps'])
            ],
            name='episode_atari'
        )
        self.log.debug('model-wide and episode summaries ok.')
        return model_summary, ep_summary


class AMLDG:
    """
    Train framework for combined model-based/model-free setup with non-parametric data model.
    Compensates model bias by jointly learning optimal policy for modelled data (generated trajectories) and
    real data model is based upon.
    Based on objective identical to one of MLDG algorithm (by Da Li et al.).

    This class is basically an AAC wrapper: it relies on two sub-AAC classes to make separate policy networks
    and training loops.

    Note that 'actor' and 'critic' names used here are not related to same named entities used in A3C and
    other actor-critic RL algorithms; it rather relevant to 'generator' and 'discriminator' terms used
    in adversarial training and mean that 'actor trainer' is optimising RL objective on synthetic data,
    generated by some model while 'critic trainer' tries to compensate model bias by optimizing same objective on
    real data model has been fitted with.

    Papers:
        Da Li et al.,
         "Learning to Generalize: Meta-Learning for Domain Generalization"
         https://arxiv.org/abs/1710.03463

        Maruan Al-Shedivat et al.,
        "Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments"
        https://arxiv.org/abs/1710.03641

    """
    def __init__(
            self,
            env,
            task,
            log_level,
            aac_class_ref=OUpAAC,
            runner_config=None,
            opt_decay_steps=None,
            opt_end_learn_rate=None,
            opt_learn_rate=1e-4,
            opt_max_env_steps=10 ** 7,
            aac_lambda=1.0,
            guided_lambda=0.0,
            tau=.5,
            rollout_length=20,
            train_phase=True,
            name='TrainAMLDG',
            **kwargs
    ):
        try:
            self.aac_class_ref = aac_class_ref
            self.task = task
            self.name = name
            self.summary_writer = None
            self.train_phase = train_phase
            self.tau = tau

            self.opt_learn_rate = opt_learn_rate
            self.opt_max_env_steps = opt_max_env_steps

            if opt_end_learn_rate is None:
                self.opt_end_learn_rate = self.opt_learn_rate
            else:
                self.opt_end_learn_rate = opt_end_learn_rate

            if opt_decay_steps is None:
                self.opt_decay_steps = self.opt_max_env_steps
            else:
                self.opt_decay_steps = opt_decay_steps

            StreamHandler(sys.stdout).push_application()
            self.log = Logger('{}_{}'.format(name, task), level=log_level)
            self.rollout_length = rollout_length

            if runner_config is None:
                self.runner_config = {
                    'class_ref': OUpRunner,
                    'kwargs': {},
                }
            else:
                self.runner_config = runner_config

            self.env_list = env

            assert isinstance(self.env_list, list) and len(self.env_list) == 2, \
                'Expected pair of environments, got: {}'.format(self.env_list)

            # Instantiate two sub-trainers: one for training on modeled data (actor, or generator) and one
            # for training on real data (critic, or discriminator):
            self.runner_config['kwargs'] = {
                'data_sample_config': {'mode': 0},  # synthetic train data
                'name': 'actor',
                'test_deterministic': not self.train_phase,
            }
            self.actor_aac = aac_class_ref(
                env=self.env_list[-1],  # test data -> slave env.
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                opt_learn_rate=self.opt_learn_rate,
                opt_max_env_steps=self.opt_max_env_steps,
                opt_end_learn_rate=self.opt_end_learn_rate,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                rollout_length=self.rollout_length,
                episode_train_test_cycle=(1, 0) if self.train_phase else (0, 1),
                _use_target_policy=False,
                _use_global_network=True,
                name=self.name + '/actor',
                **kwargs
            )
            # Change for critic:
            self.runner_config['kwargs'] = {
                'data_sample_config': {'mode': 1},  # real train data
                'name': 'critic',
                'test_deterministic': not self.train_phase,  # enable train exploration on [formally] test data
            }
            self.critic_aac = aac_class_ref(
                env=self.env_list[0],  # real train data will be master environment
                task=self.task,
                log_level=log_level,
                runner_config=self.runner_config,
                opt_learn_rate=self.opt_learn_rate,
                opt_max_env_steps=self.opt_max_env_steps,
                opt_end_learn_rate=self.opt_end_learn_rate,
                aac_lambda=aac_lambda,
                guided_lambda=guided_lambda,
                rollout_length=self.rollout_length,
                episode_train_test_cycle=(0, 1),  # always real
                _use_target_policy=False,
                _use_global_network=False,
                global_step_op=self.actor_aac.global_step,
                global_episode_op=self.actor_aac.global_episode,
                inc_episode_op=self.actor_aac.inc_episode,
                name=self.name + '/critic',
                **kwargs
            )

            self.local_steps = self.critic_aac.local_steps
            self.model_summary_freq = self.critic_aac.model_summary_freq

            self._make_train_op()

        except Exception as e:
            msg = 'AMLDG.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise e

    def _make_train_op(self):
        """
        Defines tensors holding training ops.
        """
        # Handy aliases:
        pi_critic = self.critic_aac.local_network  # local critic policy
        pi_actor = self.actor_aac.local_network  # local actor policy
        pi_global = self.actor_aac.network  # global shared policy

        # From local actor to local critic:
        self.critic_aac.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi_critic.var_list, pi_actor.var_list)]
        )

        # Inherited counters:
        self.global_step = self.actor_aac.global_step
        self.global_episode = self.actor_aac.global_episode
        self.inc_episode = self.actor_aac.inc_episode
        self.reset_global_step = self.actor_aac.reset_global_step

        # Clipped gradients for critic (critic's train op is disabled by `_use_global_network=False`
        # to avoid actor's name scope violation):
        self.critic_aac.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.critic_aac.loss, pi_critic.var_list),
            40.0
        )
        # Placeholders for stored gradients values, include None's to correctly map Vars:
        self.actor_aac.grads_placeholders = [
            tf.placeholder(shape=grad.shape, dtype=grad.dtype) if grad is not None else None
            for grad in self.actor_aac.grads
        ]
        self.critic_aac.grads_placeholders = [
            tf.placeholder(shape=grad.shape, dtype=grad.dtype) if grad is not None else None
            for grad in self.critic_aac.grads
        ]

        # Gradients to update local critic policy with stored actor's gradients:
        critic_grads_and_vars = list(zip(self.actor_aac.grads_placeholders, pi_critic.var_list))

        # Final gradients to be sent to parameter server:
        self.grads = [
            self.tau * g1 + (1 - self.tau) * g2 if g1 is not None and g2 is not None else None
            for g1, g2 in zip(self.actor_aac.grads_placeholders, self.critic_aac.grads_placeholders)
        ]
        global_grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # debug_global_grads_and_vars = list(zip(self.actor_aac.grads_placeholders, pi_global.var_list))
        # debug_global_grads_and_vars = [(g, v) for (g, v) in debug_global_grads_and_vars if g is not None]

        # Remove None entries:
        global_grads_and_vars = [(g, v) for (g, v) in global_grads_and_vars if g is not None ]
        critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars if g is not None]
        self.actor_aac.grads = [g for g in self.actor_aac.grads if g is not None]
        self.critic_aac.grads = [g for g in self.critic_aac.grads if g is not None]
        self.actor_aac.grads_placeholders = [pl for pl in self.actor_aac.grads_placeholders if pl is not None]
        self.critic_aac.grads_placeholders = [pl for pl in self.critic_aac.grads_placeholders if pl is not None]

        self.inc_step = self.actor_aac.inc_step

        # Op to update critic with gradients from actor:
        self.critic_aac.optimizer = tf.train.AdamOptimizer(self.actor_aac.learn_rate_decayed, epsilon=1e-5)
        self.update_critic_op = self.critic_aac.optimizer.apply_gradients(critic_grads_and_vars)

        # Use actor optimizer to update global policy instance:
        self.train_op = self.actor_aac.optimizer.apply_gradients(global_grads_and_vars)

        self.log.debug('all_train_ops defined')

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
            # Copy weights from global to local:
            sess.run(self.critic_aac.sync_pi)
            sess.run(self.actor_aac.sync_pi)

            # Start thread_runners:
            self.critic_aac._start_runners(   # master first
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.critic_aac.get_sample_config(mode=1)
            )
            self.actor_aac._start_runners(
                sess,
                summary_writer,
                init_context=None,
                data_sample_config=self.actor_aac.get_sample_config(mode=0)
            )

            self.summary_writer = summary_writer
            self.log.notice('Runners started.')

        except:
            msg = 'start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def process(self, sess):
        if self.train_phase:
            self.process_train(sess)

        else:
            self.process_test(sess)

    def process_test(self, sess):
        """
        Evaluation loop.
        Args:
            sess (tensorflow.Session):   tf session obj.
        """
        try:
            # sess.run(self.critic_aac.sync_pi)
            # sess.run(self.actor_aac.sync_pi)

            actor_data = self.actor_aac.get_data()
            critic_data = self.critic_aac.get_data()

            # Write down summaries:
            self.actor_aac.process_summary(sess, actor_data)
            self.critic_aac.process_summary(sess, critic_data)
            self.local_steps += 1

        except Exception as e:
            msg = 'process_test() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise e

    def process_train(self, sess):
        """
        Train procedure.

        Args:
            sess (tensorflow.Session):   tf session obj.

        """
        try:
            # Copy from parameter server:
            sess.run(self.critic_aac.sync_pi)
            sess.run(self.actor_aac.sync_pi)
            # self.log.warning('Train Sync ok.')

            # Get data configuration (redundant):
            actor_data_config = {
                'episode_config': {'get_new': 1, 'sample_type': 0, 'b_alpha': 1.0, 'b_beta': 1.0},
                'trial_config': {'get_new': 1, 'sample_type': 0, 'b_alpha': 1.0, 'b_beta': 1.0}
            }
            critic_data_config = {
                'episode_config': {'get_new': 1, 'sample_type': 1, 'b_alpha': 1.0, 'b_beta': 1.0},
                'trial_config': {'get_new': 1, 'sample_type': 1, 'b_alpha': 1.0, 'b_beta': 1.0}
            }

            # self.log.warning('actor_data_config: {}'.format(actor_data_config))
            # self.log.warning('critic_data_config: {}'.format(critic_data_config))

            # Collect synthetic train trajectory rollout:
            actor_data = self.actor_aac.get_data(data_sample_config=actor_data_config)
            actor_feed_dict = self.actor_aac.process_data(
                sess,
                actor_data,
                is_train=True,
                pi=self.actor_aac.local_network
            )

            # self.log.warning('Actor data ok.')

            wirte_model_summary = \
                self.local_steps % self.model_summary_freq == 0

            # Get gradients from actor:
            if wirte_model_summary:
                actor_fetches = [self.actor_aac.grads, self.inc_step, self.actor_aac.model_summary_op]

            else:
                actor_fetches = [self.actor_aac.grads, self.inc_step]

            # self.log.warning('self.actor_aac.grads: \n{}'.format(self.actor_aac.grads))
            # self.log.warning('self.actor_aac.model_summary_op: \n{}'.format(self.actor_aac.model_summary_op))

            actor_fetched = sess.run(actor_fetches, feed_dict=actor_feed_dict)
            actor_grads_values = actor_fetched[0]
            # self.log.warning('Actor gradients ok.')

            # Start preparing gradients feeder:
            grads_feed_dict = {
                self.actor_aac.local_network.train_phase: True,
                self.critic_aac.local_network.train_phase: True,
            }
            grads_feed_dict.update(
                {pl: value for pl, value in zip(self.actor_aac.grads_placeholders, actor_grads_values)}
            )

            # Update critic with gradients collected from generated data:
            sess.run(self.update_critic_op, feed_dict=grads_feed_dict)
            # self.log.warning('Critic update ok.')

            # Collect real train trajectory rollout using updated critic policy:
            critic_data = self.critic_aac.get_data(data_sample_config=critic_data_config)
            critic_feed_dict = self.critic_aac.process_data(
                sess,
                critic_data,
                is_train=True,
                pi=self.critic_aac.local_network
            )
            # self.log.warning('Critic data ok.')

            # Get gradients from critic:
            if wirte_model_summary:
                critic_fetches = [self.critic_aac.grads, self.critic_aac.model_summary_op]

            else:
                critic_fetches = [self.critic_aac.grads]

            critic_fetched = sess.run(critic_fetches, feed_dict=critic_feed_dict)
            critic_grads_values = critic_fetched[0]
            # self.log.warning('Critic gradients ok.')

            # Update gradients feeder with critic's:
            grads_feed_dict.update(
                {pl: value for pl, value in zip(self.critic_aac.grads_placeholders, critic_grads_values)}
            )

            # Finally send combined gradients update to parameters server:
            sess.run([self.train_op], feed_dict=grads_feed_dict)
            # sess.run([self.actor_aac.train_op], feed_dict=actor_feed_dict)

            # self.log.warning('Final gradients ok.')

            if wirte_model_summary:
                critic_model_summary = critic_fetched[-1]
                actor_model_summary = actor_fetched[-1]

            else:
                critic_model_summary = None
                actor_model_summary = None

            # Write down summaries:
            self.actor_aac.process_summary(sess, actor_data, actor_model_summary)
            self.critic_aac.process_summary(sess, critic_data, critic_model_summary)
            self.local_steps += 1

        except Exception as e:
            msg = 'process_train() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise e


class TestAMLDG(AMLDG):
    """
    Convenience evaluating wrapper.
    """
    def __init__(self, *args, train_phase=None, name=None, **kwargs):
        super().__init__(*args, train_phase=False, name='TestAMLDG', **kwargs)


class TrainAMLDG(AMLDG):
    """
    Convenience training wrapper.
    """
    def __init__(self, *args, train_phase=None, name=None, **kwargs):
        super().__init__(*args, train_phase=True, name='TrainAMLDG', **kwargs)