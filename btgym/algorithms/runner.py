# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#

import numpy as np
import six.moves.queue as queue
import threading

from btgym.algorithms import Rollout
from btgym.algorithms.memory import _DummyMemory


class RunnerThread(threading.Thread):
    """
    Async. framework code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Despite the fact BTgym is not real-time environment [yet], thread-runner approach is still here. From
    original `universe-starter-agent`:
    `...One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.`

    Another idea is to see ThreadRunner as all-in-one data provider, thus shaping data distribution
    fed to estimator from single place.
    So, replay memory is also here, as well as some service functions (collecting summary data).
    """
    def __init__(self,
                 env,
                 policy,
                 task,
                 rollout_length,
                 episode_summary_freq,
                 env_render_freq,
                 test,
                 ep_summary,
                 memory_config=None):
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
            memory_config:          replay memory configuration dictionary
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
        self.memory_config = memory_config

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
            self.ep_summary,
            self.memory_config
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
               atari_test,
               ep_summary,
               memory_config):
    """
    The logic of the thread runner.
    In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends all the collected data to the queue.

    Args:
        env:                    environment instance
        policy:                 policy instance
        task:                   int
        rollout_length:         int
        episode_summary_freq:   int
        env_render_freq:        int
        atari_test:             bool, Atari or BTGyn
        ep_summary:             dict of tf.summary op and placeholders
        memory_config:          replay memory configuration dictionary

    Yelds:
        collected data as dictionary of on_policy, off_policy rollouts and episode statistics.
    """
    if memory_config is not None:
        memory = memory_config['class_ref'](**memory_config['kwargs'])

    else:
        memory = _DummyMemory()
    # Pass sample config to environment:
    last_state = env.reset(**policy.get_sample_config())
    last_context = policy.get_initial_features(state=last_state)
    length = 0
    local_episode = 0
    reward_sum = 0
    last_action = np.zeros(env.action_space.n)
    last_action[0] = 1
    last_reward = 0.0
    last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

    # Summary averages accumulators:
    total_r = []
    cpu_time = []
    final_value = []
    total_steps = []
    total_steps_atari = []

    ep_stat = None
    test_ep_stat = None
    render_stat = None

    while True:
        terminal_end = False
        rollout = Rollout()

        action, value_, context = policy.act(last_state, last_context, last_action_reward)

        # argmax to convert from one-hot:
        state, reward, terminal, info = env.step(action.argmax())

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
        }
        # Execute user-defined callbacks to policy, if any:
        for key, callback in policy.callback.items():
            last_experience[key] = callback(**locals())

        length += 1
        reward_sum += reward
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
                #if not atari_test:
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
                memory.add(last_experience)

                # Housekeeping:
                length += 1
                reward_sum += reward
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
                total_r += [reward_sum]
                total_steps_atari += [length]
                if not atari_test:
                    episode_stat = env.get_stat()  # get episode statistic
                    last_i = info[-1]  # pull most recent info
                    cpu_time += [episode_stat['runtime'].total_seconds()]
                    final_value += [last_i['broker_value']]
                    total_steps += [episode_stat['length']]
                #print('last_episode.metadata:', state['metadata'])

                # Episode statistics:
                try:
                    # Was it test episode ( `type` in metadata is not zero)?
                    if not atari_test and state['metadata']['type']:
                        is_test_episode = True

                    else:
                        is_test_episode = False

                except KeyError:
                    is_test_episode = False

                if is_test_episode:
                    #print(task, total_r)
                    test_ep_stat = dict(
                        total_r=total_r[-1],
                        final_value=final_value[-1],
                        steps=total_steps[-1]
                    )
                else:
                    if local_episode % episode_summary_freq == 0:
                        if not atari_test:
                            # BTgym:
                            ep_stat = dict(
                                total_r=np.average(total_r),
                                cpu_time=np.average(cpu_time),
                                final_value=np.average(final_value),
                                steps=np.average(total_steps)
                            )
                        else:
                            # Atari:
                            ep_stat = dict(
                                total_r=np.average(total_r),
                                steps=np.average(total_steps_atari)
                            )
                        total_r = []
                        cpu_time = []
                        final_value = []
                        total_steps = []
                        total_steps_atari = []

                if task == 0 and local_episode % env_render_freq == 0 :
                    if not atari_test:
                        # Render environment (chief worker only, and not in atari atari_test mode):
                        render_stat = {
                            mode: env.render(mode)[None,:] for mode in env.render_modes
                        }
                    else:
                        # Atari:
                        render_stat = dict(render_atari=state['external'][None,:] * 255)

                # New episode:
                last_state = env.reset(**policy.get_sample_config())
                last_context = policy.get_initial_features(state=last_state, context=last_context)
                length = 0
                reward_sum = 0
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

        # Only training rollouts are added to replay memory:
        try:
            # Was it test (`type` in metadata is not zero)?
            if not atari_test and last_experience['state']['metadata']['type']:
                is_test = True

            else:
                is_test = False

        except KeyError:
            is_test = False

        if not is_test:
            memory.add(last_experience)

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

        # Once we have enough experience and memory can be sampled, yield it,
        # and have the ThreadRunner place it on a queue:
        if memory.is_full():
            data = dict(
                on_policy=rollout,
                off_policy=memory.sample_uniform(sequence_size=rollout_length),
                off_policy_rp=memory.sample_priority(exact_size=True),
                ep_summary=ep_stat,
                test_ep_summary=test_ep_stat,
                render_summary=render_stat,
            )
            yield data

            ep_stat = None
            test_ep_stat = None
            render_stat = None

