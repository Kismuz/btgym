import numpy as np
from logbook import Logger, StreamHandler, WARNING
import sys
import time

from btgym.algorithms.rollout import Rollout
from btgym.algorithms.memory import _DummyMemory
from btgym.algorithms.math_utils import softmax
from btgym.algorithms.utils import is_subdict


class BaseSynchroRunner():
    """
    Data provider class. Interacts with environment and outputs data in form of rollouts augmented with
    relevant summaries and metadata. This runner is `synchronous` in sense that data collection is `in-process'
    and every rollout is collected by explicit call to respective `get_data()` method [this is unlike 'async-`
    thread-runner version found earlier in this this package which, once being started,
    runs on its own and can not be moderated].
    Makes precise control on policy being executed possible.
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
            test_conditions=None,
            slowdown_steps=0,
            global_step_op=None,
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
            test:                   legacy, not used
            ep_summary:             legacy, not used
            policy:                 policy instance to execute
            data_sample_config:     dict, data sampling configuration dictionary
            memory_config:          dict, replay memory configuration
            test_conditions:        dict or None,
                                    dictionary of single experience conditions to check to mark it as test one.
            slowdown_time:          time to sleep between steps
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

        self.global_step_op = global_step_op

        if self.task == 0 and slowdown_steps > 0 and self.global_step_op is not None:
            self.log.notice('is slowed down by {} global_iterations/step'.format(slowdown_steps))
            self.slowdown_steps = slowdown_steps

        else:
            self.slowdown_steps = 0

        if test_conditions is None:
            # Default test conditions are: experience comes from test episode, from target domain:
            self.test_conditions = {
                'state': {
                    'metadata': {
                        'type': 1,
                        'trial_type': 1
                    }
                }
            }
        else:
            self.test_conditions = test_conditions

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
        self.info = [None]
        self.pre_experience = None
        self.state = None
        self.context = None
        self.action_reward = None

        # Episode accumulators:
        self.ep_accum = None

        self.norm_image = lambda x: np.round((x - x.min()) / np.ptp(x) * 255)

        self.log.debug('__init__() done.')

    def sleep(self):
        if self.slowdown_steps > 0:
            start_global_step = self.sess.run(self.global_step_op)
            while start_global_step + self.slowdown_steps > self.sess.run(self.global_step_op):
                time.sleep(0.05)

    def start_runner(self, sess, summary_writer, **kwargs):
        """
        Legacy wrapper.
        """
        self.start(sess, summary_writer, **kwargs)

    def start(self, sess, summary_writer, init_context=None, data_sample_config=None):
        """
        Executes initial sequence; fills initial replay memory if any.
        """
        assert self.policy is not None, 'Initial policy not specified'
        self.sess = sess
        self.summary_writer = summary_writer

        # Hacky but we need env.renderer methods ready:
        self.env.renderer.initialize_pyplot()

        self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
            policy=self.policy,
            init_context=init_context,
            data_sample_config=data_sample_config
        )

        if self.memory_config is not None:
            while not self.memory.is_full():
                # collect some rollouts to fill memory:
                _ = self.get_data()
            self.log.notice('Memory filled')
        self.log.notice('started collecting data.')

    def get_init_experience(self, policy, policy_sync_op=None, init_context=None, data_sample_config=None):
        """
        Starts new environment episode.

        Args:
            policy:                 policy to execute.
            policy_sync_op:         operation copying local behavioural policy params from global one
            init_context:           initial policy context for new episode.
            data_sample_config:     configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`

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

        # self.log.warning('get_init_exp() data_sample_config: {}'.format(data_sample_config))

        if data_sample_config is None:
            data_sample_config = policy.get_sample_config(**self.data_sample_config)

        # Pass sample config to environment (.get_sample_config() is actually aac framework method):
        init_state = self.env.reset(**data_sample_config)

        # Master worker always resets context at the episode beginning:
        # TODO: !
        #if not self.data_sample_config['mode']:
        init_context = None

        #self.log.warning('init_context_passed: {}'.format(init_context))
        #self.log.warning('state_metadata: {}'.format(state['metadata']))

        init_action = np.zeros(self.env.action_space.n)
        init_action[0] = 1
        init_reward = 0.0
        init_action_reward = np.concatenate([init_action, np.asarray([init_reward])], axis=-1)

        # Update policy:
        if policy_sync_op is not None:
            self.sess.run(policy_sync_op)

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
            'r': None,  # to be updated
            'info': self.info[-1],
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

        # Take a nap:
        self.sleep()

        return experience, next_state, next_context, next_action_reward

    def get_experience(self, policy, state, context, action_reward, policy_sync_op=None):
        """
        Get single experience (possibly terminal).

        Returns:
            incomplete experience as dictionary (misses bootstrapped R value),
            next_state,
            next, policy RNN context
            action_reward
        """
        # Update policy:
        if policy_sync_op is not None:
            self.sess.run(policy_sync_op)
            # self.log.warning('Per-step policy sync done.')

        # Continue adding experiences to rollout:
        action, logits, value, next_context = policy.act(state, context, action_reward)

        self.ep_accum['logits'].append(logits)
        self.ep_accum['value'].append(value)
        self.ep_accum['context'].append(context)

        # self.log.notice('context: {}'.format(context))

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
            'info': self.info[-1],
        }
        for key, callback in policy.callback.items():
            experience[key] = callback(**locals())

        # Housekeeping:
        self.length += 1

        # Take a nap:
        self.sleep()

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

    def get_ep_render(self, is_test=False):
        """
        Visualises episode environment and policy statistics.
        Relies on environmnet renderer class methods,
        so it is only valid when envoronment rendering is enabled (typically master one).

        Returns:
            dictionary of images as rgb arrays

        """
        # Only render chief worker and test (slave) environment:
        if self.task < 1 and (
            is_test or(
                self.local_episode % self.env_render_freq == 0 and not self.data_sample_config['mode']
            )
        ):

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

    def get_data(
            self,
            policy=None,
            policy_sync_op=None,
            init_context=None,
            data_sample_config=None,
            force_new_episode=False
    ):
        """
        Collects single trajectory rollout and bunch of summaries using specified policy.
        Updates episode statistics and replay memory.

        Args:
            policy:                 policy to execute
            policy_sync_op:         operation copying local behavioural policy params from global one
            init_context:           if specified, overrides initial episode context provided bu self.context
                                    (valid only if new episode is started within this rollout).
            data_sample_config:     environment configuration parameters for next episode to sample:
                                    configuration dictionary of type `btgym.datafeed.base.EnvResetConfig
            force_new_episode:      bool, if True - resets the environment


        Returns:
                data dictionary
        """
        if policy is None:
            policy = self.policy

        if init_context is None:
            init_context = self.context

        rollout = Rollout()
        is_test = False
        train_ep_summary = None
        test_ep_summary = None
        render_ep_summary = None

        if self.terminal_end or force_new_episode:
            # Start new episode:
            self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
                policy=policy,
                policy_sync_op=policy_sync_op,
                init_context=init_context,
                data_sample_config=data_sample_config
            )
            # self.log.warning(
            #     'started new episode with:\ndata_sample_config: {}\nforce_new_episode: {}'.
            #         format(data_sample_config, force_new_episode)
            # )
            # self.log.warning('pre_experience_metadata: {}'.format(self.pre_experience['state']['metadata']))

        # Collect single rollout:
        while rollout.size < self.rollout_length and not self.terminal_end:
            experience, self.state, self.context, self.action_reward = self.get_experience(
                policy=policy,
                policy_sync_op=policy_sync_op,
                state=self.state,
                context=self.context,
                action_reward=self.action_reward
            )
            # Complete previous experience by bootstrapping V from next one:
            self.pre_experience['r'] = experience['value']
            # Push:
            rollout.add(self.pre_experience)

            # Where are you coming from?
            is_test = is_subdict(self.test_conditions, self.pre_experience)

            # try:
            #     # Was it test (i.e. test episode from traget domain)?
            #     if self.pre_experience['state']['metadata']['type']\
            #             and self.pre_experience['state']['metadata']['trial_type']:
            #         is_test = True
            #
            # except KeyError:
            #     pass

            # Only training rollouts are added to replay memory:
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

                train_ep_summary = self.get_train_stat(is_test)
                test_ep_summary = self.get_test_stat(is_test)
                render_ep_summary = self.get_ep_render(is_test)

        #self.log.warning('rollout.size: {}'.format(rollout.size))

        data = dict(
            on_policy=rollout,
            terminal=self.terminal_end,
            off_policy=self.memory.sample_uniform(sequence_size=self.rollout_length),
            off_policy_rp=self.memory.sample_priority(exact_size=True),
            ep_summary=train_ep_summary,
            test_ep_summary=test_ep_summary,
            render_summary=render_ep_summary,
            is_test=is_test,
        )
        return data

    def get_episode(self, policy=None, policy_sync_op=None, init_context=None, data_sample_config=None):
        """
        WRONG
        Collects entire episode trajectory as single rollout and bunch of summaries using specified policy.
        Updates episode statistics and replay memory.

        Args:
            policy:                 policy to execute
            policy_sync_op:         operation copying local behavioural policy params from global one
            init_context:           if specified, overrides initial episode context provided bu self.context
            data_sample_config:     environment configuration parameters for next episode to sample:
                                    configuration dictionary of type `btgym.datafeed.base.EnvResetConfig

        Returns:
                data dictionary
        """
        if policy is None:
            policy = self.policy

        if init_context is None:
            init_context = self.context

        elif init_context == 0:  # mmm... TODO: fix this shame
            init_context = None

        rollout = Rollout()
        train_ep_summary = None
        test_ep_summary = None
        render_ep_summary = None

        # Start new episode:
        self.pre_experience, self.state, self.context, self.action_reward = self.get_init_experience(
            policy=policy,
            policy_sync_op=policy_sync_op,
            init_context=init_context,  # None (initially) or final context of previous episode
            data_sample_config=data_sample_config
        )

        is_test = is_subdict(self.test_conditions, self.pre_experience)
        # try:
        #     # Was it test (`type` in metadata is not zero)?
        #     # TODO: change to source/target?
        #     if self.pre_experience['state']['metadata']['type']:
        #         is_test = True
        #
        # except KeyError:
        #     pass

        # Collect data until episode is over:

        while not self.terminal_end:
            experience, self.state, self.context, self.action_reward = self.get_experience(
                policy=policy,
                policy_sync_op=policy_sync_op,
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

        train_ep_summary = self.get_train_stat(is_test)
        test_ep_summary = self.get_test_stat(is_test)
        render_ep_summary = self.get_ep_render(is_test)

        # self.log.warning('episodic_rollout.size: {}'.format(rollout.size))

        data = dict(
            on_policy=rollout,
            terminal=self.terminal_end,
            off_policy=self.memory.sample_uniform(sequence_size=self.rollout_length),
            off_policy_rp=self.memory.sample_priority(exact_size=True),
            ep_summary=train_ep_summary,
            test_ep_summary=test_ep_summary,
            render_summary=render_ep_summary,
            is_test=is_test,
        )
        return data

    def get_batch(
            self,
            size,
            policy=None,
            policy_sync_op=None,
            require_terminal=True,
            same_trial=True,
            init_context=None,
            data_sample_config=None
    ):
        """
        Returns batch as list of 'size' or more rollouts collected under specified policy.
        Rollouts can be collected from several episodes consequently; there is may be more rollouts than set 'size' if
        it is necessary to collect at least one terminal rollout.

        Args:
            size:                   int, number of rollouts to collect
            policy:                 policy to use
            policy_sync_op:         operation copying local behavioural policy params from global one
            require_terminal:       bool, if True - require at least one terminal rollout to be present.
            same_trial:             bool, if True - all episodes are sampled from same trial
            init_context:           if specified, overrides initial episode context provided bu self.context
            data_sample_config:     environment configuration parameters for all episodes in batch:
                                    configuration dictionary of type `btgym.datafeed.base.EnvResetConfig

        Returns:
            dict containing:
            'data'key holding list of data dictionaries;
            'terminal_context' key holding list of terminal output contexts.
            If 'require_terminal = True, this list is guarantied to hold at least one element.
        """

        batch = []
        terminal_context = []

        if require_terminal:
            got_terminal = False
        else:
            got_terminal = True

        if same_trial:
            assert isinstance(data_sample_config, dict),\
                'get_batch(same_trial=True) expected `data_sample_config` dict., got: {}'.format(data_sample_config)

        # Collect first rollout:
        batch = [
            self.get_data(
                policy=policy,
                policy_sync_op=policy_sync_op,
                init_context=init_context,
                data_sample_config=data_sample_config,
                force_new_episode=True
            )
        ]
        if same_trial:
            # sample new episodes from same trial only:
            data_sample_config['trial_config']['get_new'] = False

        collected_size = 1

        if batch[0]['terminal']:
            terminal_context.append(self.context)
            got_terminal = True

        # Collect others:
        while not (collected_size >= size and got_terminal):
            rollout_data = self.get_data(
                policy=policy,
                policy_sync_op=policy_sync_op,
                init_context=init_context,
                data_sample_config=data_sample_config
            )
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















