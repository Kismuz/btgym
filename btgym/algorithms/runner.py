# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#

import numpy as np
import six.moves.queue as queue
import threading

from btgym.algorithms import Rollout


class RunnerThread(threading.Thread):
    """
    Async. framework code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Despite the fact BTgym is not real-time environment [yet], thread-runner approach is still here.

    From original `universe-starter-agent`:
    ...One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, task, rollout_length, episode_summary_freq, env_render_freq, test, ep_summary):
        """

        Args:
            env:                    environment instance
            policy:                 policy instance
            task:                   int
            rollout_length:         int
            episode_summary_freq:   int
            env_render_freq:        int
            test:                   Atari or BTGyn
            ep_summary:             tf.summary
        """
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.rollout_length = rollout_length
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
        """Just keep running."""
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(
            self.sess,
            self.env,
            self.policy,
            self.task,
            self.rollout_length,
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
               rollout_length,
               summary_writer,
               episode_summary_freq,
               env_render_freq,
               test,
               ep_summary):
    """The logic of the thread runner.
    In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the rollout to the queue.

    Args:
        env:                    environment instance
        policy:                 policy instance
        task:                   int
        rollout_length:         int
        episode_summary_freq:   int
        env_render_freq:        int
        test:                   Atari or BTGyn
        ep_summary:             tf.summary

    Yelds:
        rollout instance
    """
    last_state = env.reset()
    if not test:
        last_state = last_state['model_input']

    last_context = policy.get_initial_features()
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

        action, value_, context = policy.act(last_state, last_context, last_action_reward)

        # argmax to convert from one-hot:
        state, reward, terminal, info = env.step(action.argmax())
        #if not test:
        #    state = state['model_input']

        # Partially collect first experience of rollout:
        last_experience = {
            'position': {'episode': local_episode, 'step': length},
            'state': last_state,
            'action': action,
            'reward': reward,
            'value': value_,
            'terminal': terminal,
            'context': last_context,
            'last_action_reward': last_action_reward,
            #'pixel_change': 0 #policy.get_pc_target(state, last_state),
        }
        # Execute user-defined callbacks to policy, if any:
        for key, callback in policy.callback.items():
            last_experience[key] = callback(**locals())

        length += 1
        rewards += reward
        last_state = state
        last_context = context
        last_action = action
        last_reward = reward
        last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

        for roll_step in range(1, rollout_length):
            if not terminal:
                # Continue adding experiences to rollout:
                action, value_, context = policy.act(last_state, last_context, last_action_reward)

                # Argmax to convert from one-hot:
                state, reward, terminal, info = env.step(action.argmax())
                #if not test:
                #        state = state['model_input']

                # Partially collect next experience:
                experience = {
                    'position': {'episode': local_episode, 'step': length},
                    'state': last_state,
                    'action': action,
                    'reward': reward,
                    'value': value_,
                    'terminal': terminal,
                    'context': last_context,
                    'last_action_reward': last_action_reward,
                    #'pixel_change': 0 #policy.get_pc_target(state, last_state),
                }
                for key, callback in policy.callback.items():
                    experience[key] = callback(**locals())

                # Bootstrap to complete and push previous experience:
                last_experience['r'] = value_
                rollout.add(last_experience)

                # Housekeeping:
                length += 1
                rewards += reward
                last_state = state
                last_context = context
                last_action = action
                last_reward = reward
                last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)
                last_experience = experience

            if terminal:
                # Finished episode within last taken step:
                terminal_end = True
                # All environment-specific summaries are here due to fact
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
                                ep_summary['render_atari_pl']: state['external'][None,:] * 255
                            }
                        )

                    summary_writer.add_summary(renderings, sess.run(policy.global_episode))
                    summary_writer.flush()

                # New episode:
                last_state = env.reset()
                #if not test:
                #    last_state = last_state['model_input']

                last_context = policy.get_initial_features()
                length = 0
                rewards = 0
                last_action = np.zeros(env.action_space.n)
                last_action[0] = 1
                last_reward = 0.0
                last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

                # Increment global and local episode counts:
                sess.run(policy.inc_episode)
                local_episode += 1
                break

        # After rolling `rollout_length` or less (if got `terminal`)
        # complete final experience of the rollout:
        if not terminal_end:
            # Bootstrap:
            last_experience['r'] = np.asarray(
                [policy.get_value(last_state, last_context, last_action_reward)]
            )

        else:
            last_experience['r'] = np.asarray([0.0])

        rollout.add(last_experience)

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
