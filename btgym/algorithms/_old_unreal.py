# This UNREAL implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
# https://miyosuda.github.io/
# https://github.com/miyosuda/unreal
#
# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397


from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten as flatten_nested
import six.moves.queue as queue
import threading

from btgym.algorithms import Memory, Rollout


class RunnerThread(threading.Thread):
    """
    Despite BTgym is not real-time environment [yet], thread-runner approach is still here.
    From original `universe-starter-agent`:
    ...One of the key distinctions between a normal environment and a universe environment
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
            self.num_local_steps,  # aka rollout_length
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
    runner appends the rollout to the queue.
    """
    last_state = env.reset()
    if not test:
        last_state = last_state['model_input']

    last_features = policy.get_a3c_initial_features()
    length = 0
    local_episode = 0
    rewards = 0
    last_action = np.zeros(env.action_space.n)
    last_action[0] = 1
    last_reward = 0.0
    last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

    # Summary averages accumulators:
    total_r = 0
    cpu_time = 0
    final_value = 0
    total_steps = 0
    total_steps_atari = 0

    while True:
        terminal_end = False
        rollout = Rollout()

        # Partially collect first experience of rollout:
        action, value_, features = policy.a3c_act(last_state, last_features, last_action_reward)

        # argmax to convert from one-hot:
        state, reward, terminal, info = env.step(action.argmax())
        if not test:
            state = state['model_input']
        # Estimate `pixel_change`:
        pixel_change = policy.get_pc_target(state, last_state)

        # Collect the experience:
        frame_position = {'episode': local_episode, 'step': length}
        last_experience = dict(
            position=frame_position,
            state=last_state,
            action=action,
            reward=reward,
            value=value_,
            terminal=terminal,
            features=last_features,
            pixel_change=pixel_change,
            last_action_reward=last_action_reward,  # as a[-1]
        )
        length += 1
        rewards += reward
        last_state = state
        last_features = features
        last_action = action
        last_reward = reward
        last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

        for roll_step in range(1, num_local_steps):
            if not terminal:
                # Continue adding experiences to rollout:
                action, value_, features = policy.a3c_act(last_state, last_features, last_action_reward)

                # argmax to convert from one-hot:
                state, reward, terminal, info = env.step(action.argmax())
                if not test:
                        state = state['model_input']
                pixel_change = policy.get_pc_target(state, last_state)

                # Partially collect next experience:
                frame_position = {'episode': local_episode, 'step': length}
                experience = dict(
                    position=frame_position,
                    state=last_state,
                    action=action,
                    reward=reward,
                    value=value_,
                    terminal=terminal,
                    features=last_features,
                    pixel_change=pixel_change,
                    last_action_reward=last_action_reward,
                )
                # Complete and push previous experience:
                last_experience['value_next'] = value_
                rollout.add(**last_experience)

                #print ('last_experience {}'.format(last_experience['position']))
                #for k,v in last_experience.items():
                #    try:
                #        print(k, 'shape: ', v.shape)
                #    except:
                #        try:
                #            print(k, 'type: ', type(v), 'len: ', len(v))
                #        except:
                #            print(k, 'type: ', type(v), 'value: ', v)

                #print('rollout_step: {}, last_exp/frame_pos: {}\nr: {}, v: {}, v_next: {}, t: {}'.
                #    format(
                #        length,
                #        last_experience['position'],
                #        last_experience['reward'],
                #        last_experience['value'],
                #        last_experience['value_next'],
                #        last_experience['terminal']
                #    )
                #)
                length += 1
                rewards += reward
                last_state = state
                last_features = features
                last_action = action
                last_reward = reward
                last_experience = experience

            if terminal:
                # Finished episode within last taken step:
                terminal_end = True
                #print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))

                # All environment-related summaries are here due to fact
                # only runner allowed to interact with environment:
                # Accumulate values for averaging:
                total_r += rewards
                total_steps_atari += length
                if not test:
                    episode_stat = env.get_stat()  # get episode statistic
                    last_i = info[0]  # pull most recent info
                    cpu_time += episode_stat['runtime'].total_seconds()
                    final_value += last_i['broker_value']
                    total_steps += episode_stat['length']

                # Episode statistic:
                if local_episode % episode_summary_freq == 0:
                    if not test:
                        # BTgym:

                        fetched_episode_stat = sess.run(
                            ep_summary['stat_op'],
                            feed_dict={
                                ep_summary['total_r_pl']: total_r / episode_summary_freq,
                                ep_summary['cpu_time_pl']: cpu_time / episode_summary_freq,
                                ep_summary['final_value_pl']: final_value / episode_summary_freq,
                                ep_summary['steps_pl']: total_steps / episode_summary_freq
                            }
                        )
                    else:
                        # Atari:
                        fetched_episode_stat = sess.run(
                            ep_summary['test_stat_op'],
                            feed_dict={
                                ep_summary['total_r_pl']: total_r / episode_summary_freq,
                                ep_summary['steps_pl']: total_steps_atari / episode_summary_freq
                            }
                        )
                    summary_writer.add_summary(fetched_episode_stat, sess.run(policy.global_episode))
                    summary_writer.flush()
                    total_r = 0
                    cpu_time = 0
                    final_value = 0
                    total_steps = 0
                    total_steps_atari = 0

                if task == 0 and local_episode % env_render_freq == 0 :
                    if not test:
                        # Render environment (chief worker only, and not in atari test mode):
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

                last_features = policy.get_a3c_initial_features()
                length = 0
                rewards = 0
                last_action = np.zeros(env.action_space.n)
                last_action[0] = 1
                last_reward = 0.0

                # Increment global and local episode counts:
                sess.run(policy.inc_episode)
                local_episode += 1
                break

        # After rolling `num_local_steps` or less (if got `terminal`)
        # complete final experience of the rollout:
        if not terminal_end:
            #print('last_non_terminal_value_next_added')
            last_experience['value_next'] = np.asarray(
                [policy.get_a3c_value(last_state, last_features, last_action_reward)]
            )

        else:
            #print('last_terminal_value_next_added')
            last_experience['value_next'] = np.asarray([0.0])

        rollout.add(**last_experience)

        #print('last_experience {}'.format(last_experience['position']))
        #for k, v in last_experience.items():
        #    try:
        #        print(k, 'shape: ', v.shape)
        #    except:
        #        try:
        #            print(k, 'type: ', type(v), 'len: ', len(v))
        #        except:
        #            print(k, 'type: ', type(v), 'value: ', v)

        #print('rollout_step: {}, last_exp/frame_pos: {}\nr: {}, v: {}, v_next: {}, t: {}'.
        #    format(
        #        length,
        #        last_experience['position'],
        #        last_experience['reward'],
        #        last_experience['value'],
        #        last_experience['value_next'],
        #        last_experience['terminal']
        #    )
        #)
        #print('rollout size: {}, last r: {}'.format(len(rollout.position), rollout.r[-1]))
        #print('last value_next: ', last_experience['value_next'], ', rollout flushed.')

        # Once we have enough experience, yield it, and have the ThreadRunner place it on a queue:
        yield rollout


class Unreal(object):
    """____"""
    def __init__(self,
                 env,
                 task,
                 policy_class,
                 policy_config,
                 log,
                 random_seed=0,
                 model_gamma=0.99,  # A3C decay
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
                 episode_summary_freq=2,  # every i`th episode
                 env_render_freq=10,  # every i`th episode
                 model_summary_freq=100,  # every i`th local_step
                 test_mode=False,  # gym_atari test mode
                 replay_memory_size=2000,
                 replay_rollout_length=None,
                 use_off_policy_a3c=False,
                 use_reward_prediction=False,
                 use_pixel_control=False,
                 use_value_replay=False,
                 use_rebalanced_replay=False,  # simplified form of prioritized replay
                 rebalance_skewness=2,
                 rp_lambda=1,  # aux tasks loss weights
                 pc_lambda=0.1,
                 vr_lambda=1,
                 off_a3c_lambda=1,
                 gamma_pc=0.9,  # pixel change gamma-decay - not used
                 rp_reward_threshold=0.1,  # r.prediction: abs.rewards values bigger than this are considered non-zero
                 rp_sequence_size=4,  # r.prediction sampling
                 **kwargs):
        """
        Implementation of the UNREAL algorithm.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.log = log

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.log.debug('U_{}_rnd_seed:{}, log_u_sample_(0,1]x5: {}'.
                       format(task, random_seed, self.log_uniform([1e-10,1], 5)))

        self.env = env
        self.task = task
        self.policy_class = policy_class
        self.policy_config = policy_config

        # A3C specific:
        self.model_gamma = model_gamma  # decay
        self.model_gae_lambda = model_gae_lambda  # general advantage estimator lambda
        self.model_beta = self.log_uniform(model_beta, 1)  # entropy reg.

        # Optimizer
        self.opt_max_train_steps = opt_max_train_steps
        self.opt_learn_rate = self.log_uniform(opt_learn_rate, 1)

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

        # UNREAL specific:
        self.off_a3c_lambda = off_a3c_lambda
        self.rp_lambda = rp_lambda
        self.pc_lambda = self.log_uniform(pc_lambda, 1)
        self.vr_lambda = vr_lambda
        self.gamma_pc = gamma_pc
        self.replay_memory_size = replay_memory_size
        if replay_rollout_length is not None:
            self.replay_rollout_length = replay_rollout_length
        else:
            self.replay_rollout_length = rollout_length
        self.rp_sequence_size = rp_sequence_size
        self.rp_reward_threshold = rp_reward_threshold

        # On/off switchers for off-policy training and BaseAAC auxiliary tasks:
        self.use_off_policy_a3c = use_off_policy_a3c
        self.use_reward_prediction = use_reward_prediction
        self.use_pixel_control = use_pixel_control
        if use_off_policy_a3c:
            self.use_value_replay = False  # v-replay is redundant in this case
        else:
            self.use_value_replay = use_value_replay
        self.use_rebalanced_replay = use_rebalanced_replay
        self.rebalance_skewness = rebalance_skewness

        self.use_any_aux_tasks = use_value_replay or use_pixel_control or use_reward_prediction
        self.use_memory = self.use_any_aux_tasks or self.use_off_policy_a3c

        # Make replay memory:
        self.memory = Memory( self.replay_memory_size, self.replay_rollout_length, self.rp_reward_threshold)

        self.log.info(
            'U_{}: learn_rate: {:1.6f}, entropy_beta: {:1.6f}, pc_lambda: {:1.8f}.'.
                format(self.task, self.opt_learn_rate, self.model_beta, self.pc_lambda))

        #self.log.info(
        #    'U_{}: max_steps: {}, decay_steps: {}, end_rate: {:1.6f},'.
        #        format(self.task, self.opt_max_env_steps, self.opt_decay_steps, self.opt_end_learn_rate))

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
                    self.rp_sequence_size,
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
                    self.rp_sequence_size,
                    **self.policy_config
                )
                pi.global_step = self.global_step
                pi.global_episode = self.global_episode
                pi.inc_episode = inc_episode

            # Meant for Batch-norm layers:
            pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')
            self.log.debug('U_{}: local_network_upd_ops_collection:\n{}'.format(self.task, pi.update_ops))

            self.log.debug('\nU_{}: local_network_var_list_to_save:'.format(self.task))
            for v in pi.var_list:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            # On-policy A3C loss definition:
            self.a3c_act_target = tf.placeholder(tf.float32, [None, env.action_space.n], name="a3c_action_pl")
            self.a3c_adv_target = tf.placeholder(tf.float32, [None], name="a3c_advantage_pl")
            self.a3c_r_target = tf.placeholder(tf.float32, [None], name="a3c_return_pl")

            log_prob_tf = tf.nn.log_softmax(pi.a3c_logits)
            prob_tf = tf.nn.softmax(pi.a3c_logits)  # summary only
            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that `a3c_action_target` is a placeholder that is provided externally.
            # `a3c_advantage_target` will contain the advantages, as calculated in process_rollout():

            #pi_loss = - tf.reduce_mean(
            #    tf.reduce_sum(
            #        log_prob_tf * self.a3c_act_target,
            #        [1]
            #    ) * self.a3c_adv_target
            #)

            neg_log_prob_ac = tf.nn.softmax_cross_entropy_with_logits(
                logits=pi.a3c_logits,
                labels=self.a3c_act_target
            )
            mean_neg_log_prob_ac = tf.reduce_mean(neg_log_prob_ac)
            pi_loss = tf.reduce_mean(neg_log_prob_ac * self.a3c_adv_target)

            # loss of value function:
            #vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.a3c_vf - self.a3c_r_target))

            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.a3c_vf - self.a3c_r_target))
            mean_vf = tf.reduce_mean(pi.a3c_vf)

            #entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
            entropy = tf.reduce_mean(self.cat_entropy(pi.a3c_logits))

            a3c_bs = tf.to_float(tf.shape(pi.a3c_state_in)[0])  # on-policy batch size

            a3c_loss = pi_loss + vf_loss - entropy * self.model_beta

            # Start accumulating total loss:
            self.loss = a3c_loss

            # Base summaries:
            model_summaries = [
                tf.summary.scalar("a3c/policy_loss", pi_loss),
                #tf.summary.histogram("a3c/pi_prob_d", prob_tf),
                tf.summary.scalar("a3c/value_loss", vf_loss),
                tf.summary.scalar("a3c/entropy", entropy),
                tf.summary.scalar("a3c/neg_log_pi", mean_neg_log_prob_ac),
                tf.summary.scalar("a3c/vf", mean_vf),
            ]

            # Off-policy batch size:
            off_bs = tf.to_float(tf.shape(pi.off_a3c_state_in)[0])

            if self.use_rebalanced_replay:
                # Simplified importance-sampling bias correction:
                rebalanced_replay_weight = self.rebalance_skewness / off_bs

            else:
                rebalanced_replay_weight = 1.0

            # Placeholders for off-policy training:
            self.off_policy_act_target = tf.placeholder(
                tf.float32, [None, env.action_space.n], name="off_policy_action_pl")
            self.off_policy_adv_target = tf.placeholder(
                tf.float32, [None], name="off_policy_advantage_pl")
            self.off_policy_r_target = tf.placeholder(
                tf.float32, [None], name="off_policy_return_pl")

            if self.use_off_policy_a3c:
                # Off-policy A3C loss graph mirrors on-policy:
                #off_log_prob_tf = tf.nn.log_softmax(pi.off_a3c_logits)
                #off_prob_tf = tf.nn.softmax(pi.off_a3c_logits)
                #off_pi_loss = - tf.reduce_sum(
                #    tf.reduce_sum(
                #        off_log_prob_tf * self.off_policy_action_target,
                #        [1]
                #    ) * self.off_policy_advantage_target
                #)
                off_neg_log_prob_ac = tf.nn.softmax_cross_entropy_with_logits(
                    logits=pi.off_a3c_logits,
                    labels=self.off_policy_act_target
                )
                off_pi_loss = tf.reduce_mean(off_neg_log_prob_ac * self.off_policy_adv_target)
                # loss of value function:
                off_vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.off_a3c_vf - self.off_policy_r_target))
                off_entropy = tf.reduce_mean(self.cat_entropy(pi.off_a3c_logits))
                off_a3c_loss = off_pi_loss + off_vf_loss - off_entropy * self.model_beta

                self.loss = self.loss + self.off_a3c_lambda * rebalanced_replay_weight * off_a3c_loss

                model_summaries += [
                    tf.summary.scalar("off_a3c/policy_loss", off_pi_loss),
                    tf.summary.scalar("off_a3c/value_loss", off_vf_loss),
                ]

            if self.use_pixel_control:
                # Pixel control loss
                self.pc_action = tf.placeholder(tf.float32, [None, env.action_space.n], name="pc_action")
                self.pc_target = tf.placeholder(tf.float32, [None, None, None], name="pc_target")
                # Get Q-value features for actions been taken and define loss:
                pc_action_reshaped = tf.reshape(self.pc_action, [-1, 1, 1, env.action_space.n])
                pc_q_action = tf.multiply(pi.pc_q, pc_action_reshaped)
                pc_q_action = tf.reduce_sum(pc_q_action, axis=-1, keep_dims=False)
                pc_loss = tf.reduce_sum(tf.square(self.pc_target - pc_q_action))  # sum all over

                self.loss = self.loss + self.pc_lambda * rebalanced_replay_weight * pc_loss
                # Add specific summary:
                model_summaries += [tf.summary.scalar('pixel_control/q_loss', pc_loss)]

            if self.use_value_replay:
                # Value function replay loss:
                self.vr_target = tf.placeholder(tf.float32, [None], name="vr_target")
                vr_loss = tf.reduce_mean(tf.square(pi.vr_value - self.vr_target))

                self.loss = self.loss + self.vr_lambda * rebalanced_replay_weight * vr_loss
                model_summaries += [tf.summary.scalar('v_replay/value_loss', vr_loss)]

            if self.use_reward_prediction:
                # Reward prediction loss:
                self.rp_target = tf.placeholder(tf.float32, [1,3], name="rp_target")
                rp_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.rp_target,
                        logits=pi.rp_logits
                    )
                )
                self.loss = self.loss + self.rp_lambda * rp_loss
                model_summaries += [tf.summary.scalar('r_predict/class_loss', rp_loss)]

            grads = tf.gradients(self.loss, pi.var_list)

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            self.inc_step = self.global_step.assign_add(tf.shape(pi.a3c_state_in)[0])
            #self.inc_step = self.global_step.assign_add(1)

            # Anneal learning rate:
            learn_rate = tf.train.polynomial_decay(
                self.opt_learn_rate,
                self.global_step + 1,
                self.opt_decay_steps,
                self.opt_end_learn_rate,
                power=1,
                cycle=True,
            )

            # Each worker gets a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(learn_rate)

            #opt = tf.train.RMSPropOptimizer(
            #    learning_rate=learn_rate,
            #    decay=0.99,
            #    momentum=0.0,
            #    epsilon=1e-8,
            #)

            self.train_op = tf.group(*pi.update_ops, opt.apply_gradients(grads_and_vars), self.inc_step)
            #self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # Add model-wide statistics:
            model_summaries += [
                tf.summary.scalar("global/grad_global_norm", tf.global_norm(grads)),
                tf.summary.scalar("global/var_global_norm", tf.global_norm(pi.var_list)),
                tf.summary.scalar("global/opt_learn_rate", learn_rate),
                tf.summary.scalar("global/total_loss", self.loss),
            ]

            self.summary_writer = None
            self.local_steps = 0

            self.log.debug('U_{}: train op defined'.format(self.task))

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
            # `rollout_length` represents the number of "local steps":  the number of timesteps
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
            self.log.debug('U_{}: init() done'.format(self.task))

    def log_uniform(self, lo_hi, size):
        """
        Samples from log-uniform distribution in range specified by `lo_hi`.
        Takes:
            lo_hi: either scalar or [low_value, high_value]
            size: sample size
        Returns:
             np.array or np.float (if size=1).
        """
        r = np.asarray(lo_hi)
        try:
            lo = r[0]
            hi = r[-1]
        except:
            lo = hi = r
        x = np.random.random(size)
        log_lo = np.log(lo)
        log_hi = np.log(hi)
        v = log_lo * (1 - x) + log_hi * x
        if size > 1:
            return np.exp(v)
        else:
            return np.exp(v)[0]

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

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
        """
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout
        """
        return rollout

    def process_rp(self, rp_experience_frames):
        """
        Estimates reward prediction target.
        Returns feed dictionary for `reward prediction` loss estimation subgraph.
        """
        batch_rp_state = []
        batch_rp_target = []

        for i in range(self.rp_sequence_size - 1):
            batch_rp_state.append(rp_experience_frames[i].state)

        # One hot vector for target reward (i.e. reward taken from last of sampled frames):
        r = rp_experience_frames[-1].reward
        rp_t = [0.0, 0.0, 0.0]
        if r > self.rp_reward_threshold:
            rp_t[1] = 1.0  # positive [010]

        elif r < - self.rp_reward_threshold:
            rp_t[2] = 1.0  # negative [001]

        else:
            rp_t[0] = 1.0  # zero [100]
        batch_rp_target.append(rp_t)
        feeder = {self.local_network.rp_state_in: batch_rp_state, self.rp_target: batch_rp_target}
        return feeder

    def process_vr(self, batch):
        """
        Returns feed dictionary for `value replay` loss estimation subgraph.
        """
        if not self.use_off_policy_a3c:  # use single pass of network on same off-policy batch
            feeder = {
                pl: value for pl, value in zip(self.local_network.vr_lstm_state_pl_flatten, flatten_nested(batch.features))
            }  # ...passes lstm context
            feeder.update(
                {
                    self.local_network.vr_state_in: batch.si,
                    self.local_network.vr_a_r_in: batch.last_ar,
                    #self.vr_action: batch.a,  # don't need those for value fn. estimation
                    #self.vr_advantage: batch.adv, # neither..
                    self.vr_target: batch.r,
                }
            )
        else:
            feeder = {self.vr_target: batch.r}  # redundant actually :)
        return feeder

    def process_pc(self, batch):
        """
        Returns feed dictionary for `pixel control` loss estimation subgraph.
        """
        if not self.use_off_policy_a3c:  # use single pass of network on same off-policy batch
            feeder = {
                pl: value for pl, value in zip(self.local_network.pc_lstm_state_pl_flatten, flatten_nested(batch.features))
            }
            feeder.update(
                {
                    self.local_network.pc_state_in: batch.si,
                    self.local_network.pc_a_r_in: batch.last_ar,
                    self.pc_action: batch.a,
                    self.pc_target: batch.pc
                }
            )
        else:
            feeder = {self.pc_action: batch.a, self.pc_target: batch.pc}
        return feeder

    def fill_replay_memory(self, sess):
        """
        Fills replay memory with initial experiences.
        Supposed to be called by worker() just before training begins.
        """
        if self.use_memory:
            sess.run(self.sync)
            while not self.memory.is_full():
                rollout = self.pull_batch_from_queue()
                self.memory.add_rollout(rollout)
            self.log.info('U_{}: replay memory filled.'.format(self.task))

    def process(self, sess):
        """
        Grabs a on_policy_rollout that's been produced by the thread runner,
        samples off_policy rollout[s] from replay memory and updates the parameters.
        The update is then sent to the parameter server.
        """
        sess.run(self.sync)  # copy weights from shared to local

        # Get and process on_policy_rollout for A3C train step:
        on_policy_rollout = self.pull_batch_from_queue()
        on_policy_batch = on_policy_rollout.process(gamma=self.model_gamma, gae_lambda=self.model_gae_lambda)

        # Feeder for on-policy A3C loss estimation graph:
        feed_dict = {
            pl: value for pl, value in zip(self.local_network.a3c_lstm_state_pl_flatten, flatten_nested(on_policy_batch.features))
        }  # ..passes lstm context
        feed_dict.update(
            {
                self.local_network.a3c_state_in: on_policy_batch.si,
                self.local_network.a3c_a_r_in: on_policy_batch.last_ar,
                self.a3c_act_target: on_policy_batch.a,
                self.a3c_adv_target: on_policy_batch.adv,
                self.a3c_r_target: on_policy_batch.r,
                self.local_network.train_phase: True,
            }
        )

        if self.use_off_policy_a3c or self.use_pixel_control or self.use_value_replay:
            # Get sample from replay memory:
            if self.use_rebalanced_replay:
                off_policy_sample = self.memory.sample_priority(self.replay_rollout_length,
                                                                skewness=self.rebalance_skewness, exact_size=False)
            else:
                off_policy_sample = self.memory.sample_uniform(self.replay_rollout_length)

            off_policy_rollout = Rollout()
            off_policy_rollout.add_memory_sample(off_policy_sample)
            off_policy_batch = off_policy_rollout.process(gamma=self.model_gamma, gae_lambda=self.model_gae_lambda)

            # Feeder for off-policy A3C loss estimation graph:
            off_policy_feeder = {
                pl: value for pl, value in
            zip(self.local_network.off_a3c_lstm_state_pl_flatten, flatten_nested(off_policy_batch.features))
            }
            off_policy_feeder.update(
                {
                    self.local_network.off_a3c_state_in: off_policy_batch.si,
                    self.local_network.off_a3c_a_r_in: off_policy_batch.last_ar,
                    self.off_policy_act_target: off_policy_batch.a,
                    self.off_policy_adv_target: off_policy_batch.adv,
                    self.off_policy_r_target: off_policy_batch.r,
                }
            )
            feed_dict.update(off_policy_feeder)

        # Update with reward prediction subgraph:
        if self.use_reward_prediction:
            # Rebalanced 50/50 sample for RP:
            rp_sample = self.memory.sample_priority(self.rp_sequence_size, skewness=2, exact_size=True)
            feed_dict.update(self.process_rp(rp_sample))

        # Pixel control ...
        if self.use_pixel_control:
            feed_dict.update(self.process_pc(off_policy_batch))

        # VR...
        if self.use_value_replay:
            feed_dict.update(self.process_vr(off_policy_batch))

        if self.use_memory:
            # Save on_policy_rollout to replay memory:
            self.memory.add_rollout(on_policy_rollout)

        # Every worker writes model summaries:
        should_compute_summary =\
            self.local_steps % self.model_summary_freq == 0   # self.task == 0 and

        if should_compute_summary:
            fetches = [self.model_summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        #print('TRAIN_FEED_DICT:\n', feed_dict)
        #print('\n=======S=======\n')
        #for key,value in feed_dict.items():
        #    try:
        #        print(key,':', value.shape,'\n')
        #    except:
        #        print(key, ':', value, '\n')
        #print('\n=====E======\n')

        # And finally...
        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1
