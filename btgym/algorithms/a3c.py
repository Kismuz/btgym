# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Paper: https://arxiv.org/abs/1602.01783


from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten as flatten_nested
import six.moves.queue as queue
import scipy.signal
import threading

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)
    features = rollout.features

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features = features
        #self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
    Despite BTgym is not real-time environment [yet], thread-runner approach is still here.
    From original universe-starter-agent:
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, task, num_local_steps, episode_summary_freq, env_render_freq, test, ep_summary):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq
        self.task = task
        self.test = test
        self.ep_summary = ep_summary

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(
            self.sess,
            self.env,
            self.policy,
            self.task,
            self.num_local_steps,
            self.summary_writer,
            self.episode_summary_freq,
            self.env_render_freq,
            self.test,
            self.ep_summary
        )
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(sess,
               env,
               policy,
               task,
               num_local_steps,
               summary_writer,
               episode_summary_freq,
               env_render_freq,
               test,
               ep_summary):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    if not test:
        last_state = last_state['model_input']

    last_features = policy.get_initial_features()
    length = 0
    local_episode = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()
        rollout_features = last_features

        for _ in range(num_local_steps):
            action, value_, features = policy.act(last_state, last_features)

            #test_action = np.zeros(6)
            #test_action[0] = 1
            #action = test_action

            # argmax to convert from one-hot:
            state, reward, terminal, info = env.step(action.argmax())

            #test_features = rnn.LSTMStateTuple(
            #    np.ones(features.c.shape) * reward * 10 + 10,
            #    - 1 * np.ones(features.h.shape) * reward * 10 - 10
            #)
            #features = test_features

            if not test:
                state = state['model_input']
            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, rollout_features)
            length += 1
            rewards += reward
            last_state = state
            last_features = features

            if terminal:
                terminal_end = True
                #print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))

                # All environment-related summaries are here due to fact
                # only runner allowed to interact with environment:

                # Episode statistic:
                if local_episode % episode_summary_freq == 0:
                    if not test:
                        # BTgym:
                        episode_stat = env.get_stat()  # get episode statistic
                        last_i = info[0]  # pull most recent info
                        fetched_episode_stat = sess.run(
                            ep_summary['stat_op'],
                            feed_dict={
                                ep_summary['total_r_pl']: rewards,
                                ep_summary['cpu_time_pl']: episode_stat['runtime'].total_seconds(),
                                ep_summary['final_value_pl']: last_i['broker_value'],
                                ep_summary['steps_pl']: episode_stat['length']
                            }
                        )
                    else:
                        # Atari:
                        fetched_episode_stat = sess.run(
                            ep_summary['test_stat_op'],
                            feed_dict={
                                ep_summary['total_r_pl']: rewards,
                                ep_summary['steps_pl']: length
                            }
                        )
                    summary_writer.add_summary(fetched_episode_stat, sess.run(policy.global_episode))
                    summary_writer.flush()

                if task == 0 and local_episode % env_render_freq == 0 :
                    if not test:
                        # Render environment (chief worker only, not in atari test mode):
                        #print('runner_{}: render attempt'.format(task))
                        renderings = sess.run(
                            ep_summary['render_op'],
                            feed_dict={
                                ep_summary['render_human_pl']: env.render('human')[None,:],
                                ep_summary['render_model_input_pl']: env.render('model_input')[None,:],
                                ep_summary['render_episode_pl']: env.render('episode')[None,:],
                            }
                        )
                    else:
                        # Atari:
                        # print('runner_{}: atari render attempt'.format(task))
                        renderings = sess.run(
                            ep_summary['test_render_op'],
                            feed_dict={
                                ep_summary['render_atari_pl']: state[None,:] * 255
                            }
                        )

                    summary_writer.add_summary(renderings, sess.run(policy.global_episode))
                    summary_writer.flush()

                # New episode:
                last_state = env.reset()
                if not test:
                    last_state = last_state['model_input']

                last_features = policy.get_initial_features()
                length = 0
                rewards = 0
                # Increment global and local episode counts:
                sess.run(policy.inc_episode)
                local_episode += 1
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue:
        yield rollout


class A3C(object):
    """____"""
    def __init__(self,
                 env,
                 task,
                 policy_class,
                 policy_config,
                 log,
                 model_gamma=0.99,
                 model_lambda=1.00,
                 model_beta=0.1,  # entropy regularizer
                 opt_learn_rate=1e-4,
                 opt_decay=0.99,
                 opt_momentum=0.0,
                 opt_epsilon=1e-10,
                 rollout_length=20,
                 episode_summary_freq=2,  # every i`th episode
                 env_render_freq=10,   # every i`th episode
                 model_summary_freq=100,  # every i`th local_step
                 test_mode=False,  # gym_atari test mode
                 **kwargs):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.env = env
        self.task = task
        self.policy_class = policy_class
        self.policy_config = policy_config
        self.model_gamma = model_gamma
        self.model_lambda = model_lambda
        self.model_beta = model_beta
        self.opt_learn_rate = opt_learn_rate
        self.opt_decay = opt_decay
        self.opt_epsilon = opt_epsilon
        self.opt_momentum = opt_momentum
        self.rollout_length = rollout_length
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq
        self.model_summary_freq = model_summary_freq
        self.test_mode = test_mode
        self.log = log

        self.log.debug('A3C_{}: init() started'.format(self.task))

        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        if self.test_mode:
            model_input_shape = env.observation_space.shape

        else:
            model_input_shape = env.observation_space.spaces['model_input'].shape

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = self.policy_class(
                    model_input_shape,
                    env.action_space.n,
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
                    **self.policy_config
                )
                pi.global_step = self.global_step
                pi.global_episode = self.global_episode
                pi.inc_episode = inc_episode

            # Meant for Batch-norm layers:
            pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')

            self.log.debug('A3C_{}: local_network_upd_ops_collection:\n{}'.format(self.task, pi.update_ops))

            self.log.debug('\nA3C_{}: local_network_var_list_to_save:'.format(self.task))
            for v in pi.var_list:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout:

            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function:
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])

            self.loss = pi_loss + 0.5 * vf_loss - entropy * self.model_beta

            grads = tf.gradients(self.loss, pi.var_list)

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            #
            opt = tf.train.AdamOptimizer(self.opt_learn_rate)

            #opt = tf.train.RMSPropOptimizer(
            #    learning_rate=self.opt_learn_rate,
            #    decay=0.99,
            #    momentum=0.0,
            #    epsilon=1e-8,
            #)

            self.train_op = tf.group(*pi.update_ops, opt.apply_gradients(grads_and_vars), inc_step)

            #self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            self.summary_writer = None
            self.local_steps = 0

            self.log.debug('A3C_{}: train op defined'.format(self.task))

            # Model stat. summary:
            self.model_summary_op = tf.summary.merge(
                [
                    tf.summary.scalar("model/policy_loss", pi_loss / bs),
                    tf.summary.histogram("model/a3c_logits", pi.logits),  # TEMP
                    tf.summary.scalar("model/value_loss", vf_loss / bs),
                    tf.summary.scalar("model/entropy", entropy / bs),
                    tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads)),
                    tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list)),
                    tf.summary.histogram('model/decayed_rewards_on_batch', self.r),
                ],
                name='model'
            )
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
                    tf.summary.scalar('global/episode/cpu_time_sec', self.ep_summary['cpu_time_pl']),
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

            # self.log.debug('A3C_{}: summaries ok'.format(self.task))

            # Make runner:
            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
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

            self.log.debug('A3C_{}: init() done'.format(self.task))


    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)  # starting runner thread
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)

        return rollout
        """
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout
        """

    def fill_replay_memory(self, sess):
        # unreal was here
        pass

    def process(self, sess):
        """
        Grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=self.model_gamma, lambda_=self.model_lambda)

        # Only chief worker writes model summaries:
        should_compute_summary =\
            self.local_steps % self.model_summary_freq == 0   # self.task == 0 and

        if should_compute_summary:
            fetches = [self.model_summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            pl: value for pl, value in zip(self.local_network.lstm_state_pl_flatten, flatten_nested(batch.features))
        }
        feed_dict.update(
            {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
                self.local_network.train_phase: True,
            }
        )

        #print('TRAIN_FEED_DICT:\n', feed_dict)
        #print('\n=======S=======\n')
        #for key,value in feed_dict.items():
        #    try:
        #        print(key,':', value.shape,'\n')
        #    except:
        #        print(key, ':', value, '\n')
        #print('\n=====E======\n')

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1
