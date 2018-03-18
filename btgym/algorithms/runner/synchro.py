import numpy as np
from logbook import Logger, StreamHandler, WARNING
import sys

from btgym.algorithms.rollout import Rollout
from btgym.algorithms.memory import _DummyMemory
from btgym.algorithms.math_utils import softmax


class BaseSynchroRunner():
    """
    Data provider class. Interacts with environment and outputs data in form of rollouts augmented with
    relevant summaries and metadata. This runner is `synchronous` in sense that data collection is `in-process'
    and every rollout is collected by explicit call to respective `get_data()` method [this is unlike 'async`
    thread-runner version found in this this package which, once being started,
    runs on its own and can not be moderated].
    So it makes precise control on policy being executed possible.
    Does not support 'atari' mode.
    """

    def __init__(
            self,
            env,
            task,
            rollout_length,
            episode_summary_freq,
            env_render_freq,
            ep_summary,
            test=False,
            policy=None,
            data_sample_config=None,
            memory_config=None,
            aux_summaries=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='synchro',
            log_level=WARNING,
    ):
        """

        Args:
            env:                    BTgym environment instance
            task:                   int, runner task id
            rollout_length:         int
            episode_summary_freq:   int
            env_render_freq:        int
            test:                   not used
            ep_summary:             not used
            policy:                 policy instance to execute
            data_sample_config:     dict, data sampling configuration dictionary
            memory_config:          dict, replay memory configuration
            aux_summaries:          iterable of str, additional summaries to compute
            name:                   str, name scope
            log_level:              int, logbook.level
        """
        self.env = env
        self.task = task
        self.name = name
        self.rollout_length = rollout_length
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq

        self.memory_config = memory_config
        self.policy = policy
        self.data_sample_config = data_sample_config
        self.aux_summaries = aux_summaries
        self.log_level = log_level
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('{}_Runner_{}'.format(self.name, self.task), level=self.log_level)
        self.sess = None
        self.summary_writer = None

        # Make replay memory:
        if self.memory_config is not None:
            self.memory = self.memory_config['class_ref'](**self.memory_config['kwargs'])

        else:
            self.memory = _DummyMemory()

        self.length = 0
        self.local_episode = 0
        self.reward_sum = 0

        self.terminal_end = True

        # Summary averages accumulators:
        self.total_r = []
        self.cpu_time = []
        self.final_value = []
        self.total_steps = []
        self.total_steps_atari = []
        self.info = None
        self.pre_experience = None
        self.state = None
        self.context = None
        self.action_reward = None

        # Episode accumulators:
        self.ep_accum = None

        self.norm_image = lambda x: np.round((x - x.min()) / np.ptp(x) * 255)

        self.log.debug('__init__() done.')

    def start_runner(self, sess, summary_writer):
        """Legacy wrapper"""
        self.start(sess, summary_writer)

    def start(self, sess, summary_writer):
        """
        Executes initial sequence; fills initial replay memory if set.
        """
        assert self.policy is not None, 'Initial policy not specified'
        self.sess = sess
        self.summary_writer = summary_writer

        if self.env.data_master is True:
            # Hacky but we need env.renderer methods ready
            self.env.renderer.initialize_pyplot()

        if self.memory_config is not None:
            while not self.memory.is_full():
                # collect some rollouts to fill memory:
                _ = self.get_data()
            self.log.notice('Memory filled')

        self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
            policy=self.policy,
            init_context=None
        )
        self.log.notice('started collecting data.')

    def get_init_experience(self, policy, init_context=None):
        """
        Starts new environment episode, does some housekeeping

        Args:
            init_context    initial policy context for new episode.

        Returns:
            incomplete initial experience of episode as dictionary (misses bootstrapped R value),
            next_state,
            next, policy RNN context
            action_reward
        """
        self.length = 0
        self.reward_sum = 0
        # Increment global and local episode counters:
        self.sess.run(self.policy.inc_episode)
        self.local_episode += 1

        # Pass sample config to environment (.get_sample_config() is actually aac framework method):
        init_state = self.env.reset(**policy.get_sample_config(**self.data_sample_config))

        # Master worker always resets context at the episode beginning:
        # TODO: !
        if not self.data_sample_config['mode']:
            init_context = None

        #self.log.warning('init_context_passed: {}'.format(init_context))
        #self.log.warning('state_metadata: {}'.format(state['metadata']))

        init_action = np.zeros(self.env.action_space.n)
        init_action[0] = 1
        init_reward = 0.0
        init_action_reward = np.concatenate([init_action, np.asarray([init_reward])], axis=-1)

        init_context = policy.get_initial_features(state=init_state, context=init_context)
        action, logits, value, next_context = policy.act(init_state, init_context, init_action_reward)
        next_state, reward, terminal, self.info = self.env.step(init_action.argmax())

        next_action_reward = np.concatenate([action, np.asarray([reward])], axis=-1)

        experience = {
            'position': {'episode': self.local_episode, 'step': self.length},
            'state': init_state,
            'action': action,
            'reward': reward,
            'value': value,
            'terminal': terminal,
            'context': init_context,
            'last_action_reward': init_action_reward,  # ~zeros
            'r': None  # to be updated
        }
        # Execute user-defined callbacks to policy, if any:
        for key, callback in policy.callback.items():
            experience[key] = callback(**locals())

        # reset per-episode  counters and accumulators:
        self.ep_accum = {
            'logits': [logits],
            'value': [value],
            'context': [init_context]
        }
        self.terminal_end = terminal
        #self.log.warning('init_experience_context: {}'.format(context))

        return experience, next_state, next_context, next_action_reward

    def get_experience(self, policy, state, context, action_reward):
        """
        Get single experience (possibly terminal)

        Returns:
            incomplete experience as dictionary (misses bootstrapped R value),
            next_state,
            next, policy RNN context
            action_reward
        """
        # Continue adding experiences to rollout:
        action, logits, value, next_context = policy.act(state, context, action_reward)

        # log.debug('A: {}, V: {}, step: {} '.format(action, value_, length))

        self.ep_accum['logits'].append(logits)
        self.ep_accum['value'].append(value)
        self.ep_accum['context'].append(context)

        # log.notice('context: {}'.format(context))

        # Argmax to convert from one-hot:
        next_state, reward, terminal, self.info = self.env.step(action.argmax())

        next_action_reward = np.concatenate([action, np.asarray([reward])], axis=-1)

        # Partially collect experience:
        experience = {
            'position': {'episode': self.local_episode, 'step': self.length},
            'state': state,
            'action': action,
            'reward': reward,
            'value': value,
            'terminal': terminal,
            'context': context,
            'last_action_reward': action_reward,
            'r': None,
        }
        for key, callback in policy.callback.items():
            experience[key] = callback(**locals())

        # Housekeeping:
        self.length += 1

        return experience, next_state, next_context, next_action_reward

    def get_train_stat(self, is_test=False):
        """
        Updates and computes average statistics for train episodes.
        Args:
            is_test: bool, current episode type

        Returns:
            dict of stats
        """
        ep_stat = {}
        if not is_test:
            self.total_r += [self.reward_sum]
            episode_stat = self.env.get_stat()  # get episode statistic
            last_i = self.info[-1]  # pull most recent info
            self.cpu_time += [episode_stat['runtime'].total_seconds()]
            self.final_value += [last_i['broker_value']]
            self.total_steps += [episode_stat['length']]

        if self.local_episode % self.episode_summary_freq == 0:
            ep_stat = dict(
                total_r=np.average(self.total_r),
                cpu_time=np.average(self.cpu_time),
                final_value=np.average(self.final_value),
                steps=np.average(self.total_steps)
            )
            self.total_r = []
            self.cpu_time = []
            self.final_value = []
            self.total_steps = []
            self.total_steps_atari = []

        return ep_stat

    def get_test_stat(self, is_test=False):
        """
        Updates and computes  statistics for single test episode.

        Args:
            is_test: bool, current episode type

        Returns:
            dict of stats

        """
        ep_stat = {}
        if is_test:
            episode_stat = self.env.get_stat()  # get episode statistic
            last_i = self.info[-1]  # pull most recent info
            ep_stat = dict(
                total_r=self.reward_sum,
                final_value=last_i['broker_value'],
                steps=episode_stat['length']
            )
        return ep_stat

    def get_ep_render(self):
        """
        Visualises episode environment and policy statistics.

        Returns:
            dictionary of images as rgb arrays

        """
        # Only render chief worker and test environment:
        # TODO: train as well; source/target?
        if self.task < 1 and self.data_sample_config['mode'] and self.local_episode % self.env_render_freq == 0:
            # TODO:  !
        # if self.task < 1  and self.local_episode % self.env_render_freq == 0:

            # Render environment (chief worker only):
            render_stat = {
                mode: self.env.render(mode)[None, :] for mode in self.env.render_modes
            }
            # Update renderings with aux:

            # log.notice('ep_logits shape: {}'.format(np.asarray(ep_a_logits).shape))
            # log.notice('ep_value shape: {}'.format(np.asarray(ep_value).shape))

            # Unpack LSTM states:
            rnn_1, rnn_2 = zip(*self.ep_accum['context'])
            rnn_1 = [state[0] for state in rnn_1]
            rnn_2 = [state[0] for state in rnn_2]
            c1, h1 = zip(*rnn_1)
            c2, h2 = zip(*rnn_2)

            aux_images = {
                'action_prob': self.env.renderer.draw_plot(
                    # data=softmax(np.asarray(ep_a_logits)[:, 0, :] - np.asarray(ep_a_logits).max()),
                    data=softmax(np.asarray(self.ep_accum['logits'])[:, 0, :]),
                    title='Episode actions probabilities',
                    figsize=(12, 4),
                    box_text='',
                    xlabel='Backward env. steps',
                    ylabel='R+',
                    line_labels=['Hold', 'Buy', 'Sell', 'Close']
                )[None, ...],
                'value_fn': self.env.renderer.draw_plot(
                    data=np.asarray(self.ep_accum['value']),
                    title='Episode Value function',
                    figsize=(12, 4),
                    xlabel='Backward env. steps',
                    ylabel='R',
                    line_labels=['Value']
                )[None, ...],
                # 'lstm_1_c': norm_image(np.asarray(c1).T[None, :, 0, :, None]),
                'lstm_1_h': self.norm_image(np.asarray(h1).T[None, :, 0, :, None]),
                # 'lstm_2_c': norm_image(np.asarray(c2).T[None, :, 0, :, None]),
                'lstm_2_h': self.norm_image(np.asarray(h2).T[None, :, 0, :, None])
            }
            render_stat.update(aux_images)

        else:
            render_stat = None

        return render_stat

    def get_data(self, policy=None):
        """
        Collects data consisting of single rollout and bunch of summaries using specified policy.
        Updates episode statistics and replay memory.

        Args:
            policy:     policy to execute

        Returns:
                data dictionary
        """
        if policy is None:
            policy = self.policy

        rollout = Rollout()
        is_test = False
        train_ep_summary = None
        test_ep_summary = None
        render_ep_summary = None

        if self.terminal_end:
            # Start new episode:
            self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
                policy=policy,
                init_context=self.context  # None (initially) or final context of previous episode
            )

        # Collect single rollout:
        while rollout.size < self.rollout_length and not self.terminal_end:
            experience, self.state, self.context, self.action_reward = self.get_experience(
                policy=policy,
                state=self.state,
                context=self.context,
                action_reward=self.action_reward
            )
            # Complete previous experience by bootstrapping V from next one:
            self.pre_experience['r'] = experience['value']
            # Push:
            rollout.add(self.pre_experience)

            # Only training rollouts are added to replay memory:
            is_test = False
            try:
                # Was it test (`type` in metadata is not zero)?
                # TODO: change to source/target?
                if self.pre_experience['state']['metadata']['type']:
                    is_test = True

            except KeyError:
                pass

            if not is_test:
                self.memory.add(self.pre_experience)

            # Move one step froward:
            self.pre_experience = experience

            self.reward_sum += experience['reward']

            if self.pre_experience['terminal']:
                # Episode has been just finished,
                # need to complete and push last experience and update all episode summaries:
                self.terminal_end = True

                # Bootstrap:
                self.pre_experience['r'] = np.asarray(
                    [
                        policy.get_value(
                            self.pre_experience['state'],
                            self.pre_experience['context'],
                            self.pre_experience['last_action_reward']
                        )
                    ]
                )
                rollout.add(self.pre_experience)
                if not is_test:
                    self.memory.add(self.pre_experience)

                #train_ep_summary = self.get_train_stat(is_test)
                train_ep_summary = self.get_train_stat(False)
                #test_ep_summary = self.get_test_stat(is_test)
                render_ep_summary = self.get_ep_render()

        #self.log.warning('rollout.size: {}'.format(rollout.size))

        data = dict(
            on_policy=rollout,
            terminal=self.terminal_end,
            off_policy=self.memory.sample_uniform(sequence_size=self.rollout_length),
            off_policy_rp=self.memory.sample_priority(exact_size=True),
            ep_summary=train_ep_summary,
            test_ep_summary=test_ep_summary,
            render_summary=render_ep_summary,
        )
        return data

    def get_episode(self, policy=None, init_context=None):
        """
        Collects entire episode trajectory and bunch of summaries using specified policy.
        Updates episode statistics and replay memory.

        Args:
            policy:     policy to execute

        Returns:
                data dictionary
        """
        if policy is None:
            policy = self.policy

        if init_context is None:
            init_context = self.context
        elif init_context == 0:
            init_context = None

        rollout = Rollout()
        train_ep_summary = None
        test_ep_summary = None
        render_ep_summary = None

        # Start new episode:
        self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
            policy=policy,
            init_context=init_context  # None (initially) or final context of previous episode
        )
        # Only training rollouts are added to replay memory:
        is_test = False
        try:
            # Was it test (`type` in metadata is not zero)?
            # TODO: change to source/target?
            if self.pre_experience['state']['metadata']['type']:
                is_test = True

        except KeyError:
            pass

        # Collect data until episode is over:

        while not self.terminal_end:
            experience, self.state, self.context, self.action_reward = self.get_experience(
                policy=policy,
                state=self.state,
                context=self.context,
                action_reward=self.action_reward
            )
            # Complete previous experience by bootstrapping V from next one:
            self.pre_experience['r'] = experience['value']
            # Push:
            rollout.add(self.pre_experience)

            if not is_test:
                self.memory.add(self.pre_experience)

            # Move one step froward:
            self.pre_experience = experience

            self.reward_sum += experience['reward']

            if self.pre_experience['terminal']:
                # Episode has been just finished,
                # need to complete and push last experience and update all episode summaries:
                self.terminal_end = True

        # Bootstrap:
        self.pre_experience['r'] = np.asarray(
            [
                policy.get_value(
                    self.pre_experience['state'],
                    self.pre_experience['context'],
                    self.pre_experience['last_action_reward']
                )
            ]
        )
        rollout.add(self.pre_experience)
        if not is_test:
            self.memory.add(self.pre_experience)

        # train_ep_summary = self.get_train_stat(is_test)
        train_ep_summary = self.get_train_stat(False)
        # test_ep_summary = self.get_test_stat(is_test)
        render_ep_summary = self.get_ep_render()

        # self.log.warning('episodic_rollout.size: {}'.format(rollout.size))

        data = dict(
            on_policy=rollout,
            terminal=self.terminal_end,
            off_policy=self.memory.sample_uniform(sequence_size=self.rollout_length),
            off_policy_rp=self.memory.sample_priority(exact_size=True),
            ep_summary=train_ep_summary,
            test_ep_summary=test_ep_summary,
            render_summary=render_ep_summary,
        )
        return data

    def get_batch(self, policy, size, require_terminal=True):
        """
        Returns batch of 'size' or more rollouts collected under specified policy.
        Rollouts can be collected from several episodes consequently.

        Args:
            policy:             policy to use
            size:               int, number of rollouts to collect
            require_terminal:   bool, if True - require at least one terminal rollout to be present.

        Returns:
            dict containing: list of data dictionaries; 'terminal_context' key holding list of terminal
            output contexts. If 'require_terminal = True, this list is guarantied to hold at least one value.
        """
        collected_size = 0
        batch = []
        terminal_context = []

        if require_terminal:
            got_terminal = False
        else:
            got_terminal = True

        # Collect rollouts:
        while not collected_size >= size and got_terminal:
            rollout_data = self.get_data(policy)
            batch.append(rollout_data)

            if rollout_data['terminal']:
                terminal_context.append(self.context)
                got_terminal = True

            collected_size += 1

        data = dict(
            data=batch,
            terminal_context=terminal_context,
        )
        return data















