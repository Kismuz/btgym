
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten as flatten_nested

from btgym.algorithms import RunnerThread

from btgym.algorithms.math_util import log_uniform
from btgym.algorithms.losses import aac_loss_def

class A3C(object):
    """ Asynchronous Advantage Actor Critic.

    Original code is taken from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Paper: https://arxiv.org/abs/1602.01783
"""
    def __init__(self,
                 env,
                 task,
                 policy_class,
                 policy_config,
                 log,
                 random_seed=None,
                 model_gamma=0.99,  # decay
                 model_gae_lambda=1.00,  # GAE lambda
                 model_beta=0.01,  # entropy regularizer
                 opt_max_train_steps=10**7,
                 opt_decay_steps=None,
                 opt_end_learn_rate=None,
                 opt_learn_rate=1e-4,
                 opt_decay=0.99,
                 opt_momentum=0.0,
                 opt_epsilon=1e-10,
                 rollout_length=20,
                 episode_summary_freq=2,  # every i`th environment episode
                 env_render_freq=10,  # every i`th environment episode
                 model_summary_freq=100,  # every i`th algorithm iteration
                 test_mode=False,  # gym_atari test mode
                 **kwargs):
        """

        Args:
            env:                    envirionment instance.
            task:                   int
            policy_class:           policy estimator class
            policy_config:          config dictionary
            log:                    parent log
            random_seed:            int or None
            model_gamma:            gamma discount factor
            model_gae_lambda:       GAE lambda
            model_beta:             entropy regularization beta
            opt_max_train_steps:    train steps to run
            opt_decay_steps:        learn ratio decay steps
            opt_end_learn_rate:     final lerarn rate
            opt_learn_rate:         start learn rate
            opt_decay:              optimizer decay, if apll.
            opt_momentum:           optimizer momentum, if apll.
            opt_epsilon:            optimizer epsilon
            rollout_length:         on-policy rollout length
            episode_summary_freq:   int, write episode summary for every i'th episode
            env_render_freq:        int, write environment rendering summary for every i'th train step
            model_summary_freq:     int, write model summary for every i'th train step
            test_mode:              True: Atari, False: BTGym
            **kwargs:               NOT USED
        """
        self.log = log
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)
        self.log.debug('AAC_{}_rnd_seed:{}, log_u_sample_(0,1]x5: {}'.
                       format(task, random_seed, log_uniform([1e-10,1], 5)))

        self.env = env
        self.task = task
        self.policy_class = policy_class
        self.policy_config = policy_config

        # AAC specific:
        self.model_gamma = model_gamma  # decay
        self.model_gae_lambda = model_gae_lambda  # general advantage estimator lambda
        self.model_beta = log_uniform(model_beta, 1)  # entropy reg.

        # Optimizer
        self.opt_max_train_steps = opt_max_train_steps
        self.opt_learn_rate = log_uniform(opt_learn_rate, 1)

        if opt_end_learn_rate is None:
            self.opt_end_learn_rate = self.opt_learn_rate
        else:
            self.opt_end_learn_rate = opt_end_learn_rate

        if opt_decay_steps is None:
            self.opt_decay_steps = self.opt_max_train_steps
        else:
            self.opt_decay_steps = opt_decay_steps

        self.opt_decay = opt_decay
        self.opt_epsilon = opt_epsilon
        self.opt_momentum = opt_momentum
        self.rollout_length = rollout_length

        # Summaries :
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq
        self.model_summary_freq = model_summary_freq

        # If True - use ATARI gym env.:
        self.test_mode = test_mode

        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        if self.test_mode:
            model_input_shape = env.observation_space.shape

        else:
            model_input_shape = env.observation_space.spaces['model_input'].shape

        # Start building graph:
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = self.policy_class(
                    model_input_shape,
                    env.action_space.n,
                    3,
                    **self.policy_config
                )
                self.global_step = tf.get_variable(
                    "global_step",
                    [],
                    tf.int32,
                    initializer=tf.constant_initializer(
                        0,
                        dtype=tf.int32
                    ),
                    trainable=False
                )
                self.global_episode = tf.get_variable(
                    "global_episode",
                    [],
                    tf.int32,
                    initializer=tf.constant_initializer(
                        0,
                        dtype=tf.int32
                    ),
                    trainable=False
                )
        # Increment episode count:
        inc_episode = self.global_episode.assign_add(1)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = self.policy_class(
                    model_input_shape,
                    env.action_space.n,
                    3,
                    **self.policy_config
                )
                pi.global_step = self.global_step
                pi.global_episode = self.global_episode
                pi.inc_episode = inc_episode

            # Meant for Batch-norm layers:
            pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')
            self.log.debug('AAC_{}: local_network_upd_ops_collection:\n{}'.format(self.task, pi.update_ops))

            self.log.debug('\nAAC_{}: local_network_var_list_to_save:'.format(self.task))
            for v in pi.var_list:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            #  Learning rate annealing:
            learn_rate = tf.train.polynomial_decay(
                self.opt_learn_rate,
                self.global_step + 1,
                self.opt_decay_steps,
                self.opt_end_learn_rate,
                power=1,
                cycle=False,
            )

            # On-policy AAC loss definition:
            self.on_pi_act_target = tf.placeholder(tf.float32, [None, env.action_space.n], name="on_policy_action_pl")
            self.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            self.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            self.loss, model_summaries = aac_loss_def(
                act_target=self.on_pi_act_target,
                adv_target=self.on_pi_adv_target,
                r_target=self.on_pi_r_target,
                pi_logits=pi.on_logits,
                pi_vf=pi.on_vf,
                entropy_beta=self.model_beta,
                name='a3c',
                verbose=True
            )
            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # Copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in)[0])

            # Each worker gets a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(learn_rate, epsilon=1e-5)

            #opt = tf.train.RMSPropOptimizer(
            #    learning_rate=learn_rate,
            #    decay=0.99,
            #    momentum=0.0,
            #    epsilon=1e-8,
            #)

            self.train_op = opt.apply_gradients(grads_and_vars)

            # Add model-wide statistics:
            with tf.name_scope('model'):
                model_summaries += [
                    tf.summary.scalar("grad_global_norm", tf.global_norm(grads)),
                    tf.summary.scalar("var_global_norm", tf.global_norm(pi.var_list)),
                    tf.summary.scalar("learn_rate", learn_rate),
                    tf.summary.scalar("total_loss", self.loss),
                ]

            self.summary_writer = None
            self.local_steps = 0

            self.log.debug('AAC_{}: train op defined'.format(self.task))

            # Model stat. summary:
            self.model_summary_op = tf.summary.merge(model_summaries, name='model_summary')

            # Episode-related summaries:
            self.ep_summary = dict(
                # Summary placeholders
                render_human_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_model_input_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_episode_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_atari_pl=tf.placeholder(tf.uint8, [None, None, None, 1]),
                total_r_pl=tf.placeholder(tf.float32, ),
                cpu_time_pl=tf.placeholder(tf.float32, ),
                final_value_pl=tf.placeholder(tf.float32, ),
                steps_pl=tf.placeholder(tf.int32, ),
            )
            # Environmnet rendering:
            self.ep_summary['render_op'] = tf.summary.merge(
                [
                    tf.summary.image('human', self.ep_summary['render_human_pl']),
                    tf.summary.image('model_input', self.ep_summary['render_model_input_pl']),
                    tf.summary.image('episode', self.ep_summary['render_episode_pl']),
                ],
                name='render'
            )
            # For Atari:
            self.ep_summary['test_render_op'] = tf.summary.image("model/state", self.ep_summary['render_atari_pl'])

            # Episode stat. summary:
            self.ep_summary['stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode/total_reward', self.ep_summary['total_r_pl']),
                    tf.summary.scalar('episode/cpu_time_sec', self.ep_summary['cpu_time_pl']),
                    tf.summary.scalar('episode/final_value', self.ep_summary['final_value_pl']),
                    tf.summary.scalar('episode/env_steps', self.ep_summary['steps_pl'])
                ],
                name='episode'
            )
            self.ep_summary['test_stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode/total_reward', self.ep_summary['total_r_pl']),
                    tf.summary.scalar('episode/steps', self.ep_summary['steps_pl'])
                ],
                name='episode_atari'
            )

            # Make runner:
            self.runner = RunnerThread(
                env,
                pi,
                task,
                self.rollout_length,  # ~20
                self.episode_summary_freq,
                self.env_render_freq,
                self.test_mode,
                self.ep_summary
            )
            self.log.debug('AAC_{}: init() done'.format(self.task))

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)  # starting runner thread
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        #self.log.debug('Rollout position:{}\nactions:{}\nrewards:{}\nlast_action:{}\nlast_reward:{}\nterminal:{}\n'.
        #      format(rollout.position, rollout.actions,
        #             rollout.rewards, rollout.last_actions, rollout.last_rewards, rollout.terminal))
        return rollout

    def process(self, sess):
        """
        Grabs a on_policy_rollout that's been produced by the thread runner,
        samples off_policy rollout[s] from replay memory and updates the parameters.
        The update is then sent to the parameter server.
        """

        # Copy weights from shared to local new_policy:
        sess.run(self.sync)


        # Get and process rollout:
        on_policy_rollout = self.pull_batch_from_queue()
        on_policy_batch = on_policy_rollout.process(gamma=self.model_gamma, gae_lambda=self.model_gae_lambda)

        # Feeder for on-policy AAC loss estimation graph:
        feed_dict = {pl: value for pl, value in
                     zip(self.local_network.on_lstm_state_pl_flatten, flatten_nested(on_policy_batch['context']))}

        feed_dict.update(
            {
                self.local_network.on_state_in: on_policy_batch['state'],
                self.local_network.on_a_r_in: on_policy_batch['last_action_reward'],
                self.on_pi_act_target: on_policy_batch['action'],
                self.on_pi_adv_target: on_policy_batch['advantage'],
                self.on_pi_r_target: on_policy_batch['r'],
                self.local_network.train_phase: True,
            }
        )

        # Every worker writes model summaries:
        should_compute_summary =\
            self.local_steps % self.model_summary_freq == 0

        if should_compute_summary:
            fetches = [self.train_op, self.model_summary_op, self.inc_step]
        else:
            fetches = [self.train_op, self.inc_step]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1

        #for k, v in feed_dict.items():
        #    try:
        #        print(k, v.shape)
        #    except:
        #        print(k, type(v))