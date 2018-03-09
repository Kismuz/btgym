
import numpy as np

from btgym.algorithms.rollout import Rollout
from btgym.algorithms.memory import _DummyMemory
from btgym.algorithms.math_utils import softmax


def MetaEnvRunnerFn(
        sess,
        env,
        policy,
        task,
        rollout_length,
        summary_writer,
        episode_summary_freq,
        env_render_freq,
        atari_test,
        ep_summary,
        memory_config,
        log,
        aux_summaries=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
):
    """
    Meta-learning loop runtime logic of the thread runner.

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
        log:                    logbook logger
        aux_summaries:          list of str, additional summaries to compute

    Yelds:
        collected data as dictionary of on_policy, off_policy rollouts, episode statistics and summaries.
    """
    if memory_config is not None:
        memory = memory_config['class_ref'](**memory_config['kwargs'])

    else:
        memory = _DummyMemory()

    # We want test data runner to be master:

    if 'test' in task:
        mode = 1
    else:
        mode = 0
    # Pass sample config to environment (.get_sample_config() is actually aac framework method):
    log.warning('mode={}'.format(mode))
    last_state = env.reset(**policy.get_sample_config(mode))
    last_action, last_reward, last_value, last_context = policy.get_initial_features(state=last_state)
    length = 0
    local_episode = 0
    reward_sum = 0
    last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

    # Summary averages accumulators:
    total_r = []
    cpu_time = []
    final_value = []
    total_steps = []
    total_steps_atari = []

    # Aux accumulators:
    ep_a_logits = []
    ep_value = []
    ep_context = []

    ep_stat = None
    test_ep_stat = None
    render_stat = None

    norm_image = lambda x: np.round((x - x.min()) / np.ptp(x) * 255)

    if env.data_master is True:
        # Hacky but we need env.renderer methods ready
        env.renderer.initialize_pyplot()

    log.notice('started data collection.')
    while True:
        terminal_end = False
        rollout = Rollout()

        action, logits, value, context = policy.act(last_state, last_context, last_action_reward)

        ep_a_logits.append(logits)
        ep_value.append(value)
        ep_context.append(context)

        #log.debug('*: A: {}, V: {}, step: {} '.format(action, value_, length))

        # argmax to convert from one-hot:
        state, reward, terminal, info = env.step(action.argmax())

        # Partially collect first experience of rollout:
        last_experience = {
            'position': {'episode': local_episode, 'step': length},
            'state': last_state,
            'action': action,
            'reward': reward,
            'value': value,
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
                action, logits, value, context = policy.act(last_state, last_context, last_action_reward)

                #log.debug('A: {}, V: {}, step: {} '.format(action, value_, length))

                ep_a_logits.append(logits)
                ep_value.append(value)
                ep_context.append(context)

                #log.notice('context: {}'.format(context))

                # Argmax to convert from one-hot:
                state, reward, terminal, info = env.step(action.argmax())

                # Partially collect next experience:
                experience = {
                    'position': {'episode': local_episode, 'step': length},
                    'state': last_state,
                    'action': action,
                    'reward': reward,
                    'value': value,
                    'terminal': terminal,
                    'context': last_context,
                    'last_action_reward': last_action_reward,
                    #'pixel_change': 0 #policy.get_pc_target(state, last_state),
                }
                for key, callback in policy.callback.items():
                    experience[key] = callback(**locals())

                # Bootstrap to complete and push previous experience:
                last_experience['r'] = value
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
                # Only render chief worker and test environment:
                if '0' in task and 'test' in task and local_episode % env_render_freq == 0 :
                    if not atari_test:
                        # Render environment (chief worker only, and not in atari atari_test mode):
                        render_stat = {
                            mode: env.render(mode)[None,:] for mode in env.render_modes
                        }
                        # Update renderings with aux:

                        # log.notice('ep_logits shape: {}'.format(np.asarray(ep_a_logits).shape))
                        # log.notice('ep_value shape: {}'.format(np.asarray(ep_value).shape))

                        # Unpack LSTM states:
                        rnn_1, rnn_2 = zip(*ep_context)
                        rnn_1 = [state[0] for state in rnn_1]
                        rnn_2 = [state[0] for state in rnn_2]
                        c1, h1 = zip(*rnn_1)
                        c2, h2 = zip(*rnn_2)

                        aux_images = {
                            'action_prob':  env.renderer.draw_plot(
                                # data=softmax(np.asarray(ep_a_logits)[:, 0, :] - np.asarray(ep_a_logits).max()),
                                data=softmax(np.asarray(ep_a_logits)[:, 0, :]),
                                title='Episode actions probabilities',
                                figsize=(12, 4),
                                box_text='',
                                xlabel='Backward env. steps',
                                ylabel='R+',
                                line_labels=['Hold', 'Buy', 'Sell', 'Close']
                            )[None, ...],
                            'value_fn': env.renderer.draw_plot(
                                data=np.asarray(ep_value),
                                title='Episode Value function',
                                figsize=(12, 4),
                                xlabel='Backward env. steps',
                                ylabel='R',
                                line_labels = ['Value']
                            )[None, ...],
                            #'lstm_1_c': norm_image(np.asarray(c1).T[None, :, 0, :, None]),
                            'lstm_1_h': norm_image(np.asarray(h1).T[None, :, 0, :, None]),
                            #'lstm_2_c': norm_image(np.asarray(c2).T[None, :, 0, :, None]),
                            'lstm_2_h': norm_image(np.asarray(h2).T[None, :, 0, :, None])
                        }

                        render_stat.update(aux_images)

                    else:
                        # Atari:
                        render_stat = dict(render_atari=state['external'][None,:] * 255)

                # New episode:
                last_state = env.reset(**policy.get_sample_config(mode))
                last_action, last_reward, last_value, last_context = policy.get_initial_features(state=last_state)
                length = 0
                reward_sum = 0
                last_action_reward = np.concatenate([last_action, np.asarray([last_reward])], axis=-1)

                # reset per-episode accumulators:
                ep_a_logits = []
                ep_value = []
                ep_context = []

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

