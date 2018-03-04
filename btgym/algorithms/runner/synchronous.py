import numpy as np
from logbook import Logger, StreamHandler, WARNING
import sys

from btgym.algorithms.rollout import Rollout
from btgym.algorithms.memory import _DummyMemory
from btgym.algorithms.math_utils import softmax

class BaseSynchroRunner():
    """
    Data provider class. Interacts with environment and outputs data in form of rollouts packed
    with relevant metadata. This runner is `synchronous` in sense that data collection is `in-process'
    and controlled by explicit call to respective method [this is unlike 'async` thread-runner version of this package
    which, once being started, runs on its owne]. This allows precise control on policy being executed by runner.
    """

    def __init__(
            self,
            env,
            task,
            rollout_length,
            episode_summary_freq,
            env_render_freq,
            test,
            ep_summary,
            policy=None,
            memory_config=None,
            aux_summaries=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            log_level=WARNING,
    ):
        """

        Args:
            env:                    BTgym environment instance
            task:                   str, runner id
            rollout_length:         int
            episode_summary_freq:   int
            env_render_freq:        int
            test:                   bool, Atari or BTgym
            ep_summary:             tf.summary
            policy:                 policy instance to execute
            memory_config:          dict, replay memory configuration
            aux_summaries:          iterable of str, additional summaries to compute
            log_level:              int, logbook.level
        """
        self.env = env
        self.task = task
        self.rollout_length = rollout_length
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq
        self.test = test
        self.ep_summary = ep_summary
        self.memory_config = memory_config
        self.policy = policy
        self.aux_summaries = aux_summaries
        self.log_level = log_level
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('Runner_{}'.format(self.task), level=self.log_level)
        self.sess = None
        self.summary_writer = None

        # Make replay memory:
        if self.memory_config is not None:
            self.memory = self.memory_config['class_ref'](**self.memory_config['kwargs'])

        else:
            self.memory = _DummyMemory()

        # We want `test` [or T_i+1] data runner to be master:
        if 'test' in self.task:
            self.mode = 1
        else:
            self.mode = 0

        self.length = 0
        self.local_episode = 0
        self.reward_sum = 0

        self.last_state = None
        self.last_action_reward = None
        self.last_action = None
        self.last_reward = None
        self.last_value = None
        self.last_context = None

        # Summary averages accumulators:
        self.total_r = []
        self.cpu_time = []
        self.final_value = []
        self.total_steps = []
        self.total_steps_atari = []

        # Aux accumulators:
        self.ep_a_logits = []
        self.ep_value = []
        self.ep_context = []

        self.ep_stat = None
        self.test_ep_stat = None
        self.render_stat = None

        self.norm_image = lambda x: np.round((x - x.min()) / np.ptp(x) * 255)

    def start_runner(self, **kwargs):
        """Legacy wrapper"""
        self.start(**kwargs)

    def start(self, sess, summary_writer):
        """
        Executes initial sequence; fills initial replay memory if any.
        """
        assert self.policy is not None, 'Initial policy not specified'
        self.sess = sess
        self.summary_writer = summary_writer

        if self.env.data_master is True:
            # Hacky but we need env.renderer methods ready
            self.env.renderer.initialize_pyplot()

        self.terminal_end = False
        self.rollout = Rollout()

        self.log.notice('Ready?')

        self.last_experience = self._new_episode()

        # TODO: fill replay memory first!

    def _new_episode(self, init_context=None):
        """
        Starts new environment episode and does relevant housekeeping.

        Args:
            init_context    initial policy context for new episode.

        Returns:
            incomplete experience as dictionary
        """
        self.length = 0
        self.reward_sum = 0
        # Increment global and local episode counters:
        self.sess.run(self.policy.inc_episode)
        self.local_episode += 1

        # Pass sample config to environment (.get_sample_config() is actually aac framework method):
        state = self.env.reset(**self.policy.get_sample_config(self.mode))
        context = self.policy.get_initial_features(state=state, context=init_context)

        action = np.zeros(self.env.action_space.n)
        action[0] = 1
        reward = 0.0
        last_action_reward = np.concatenate([action, np.asarray([reward])], axis=-1)


        experience = {
            'position': {'episode': self.local_episode, 'step': self.length},
            'state': state,
            'action': action,
            'reward': reward,
            'value': value,
            'terminal': False,
            'context': context,
            'last_action_reward': last_action_reward,
            'r': None  # to be updated
        }
        # Execute user-defined callbacks to policy, if any:
        for key, callback in self.policy.callback.items():
            experience[key] = callback(**locals())

        # reset per-episode  counters and accumulators:

        self.ep_a_logits = [action]  # kinda logits :/
        self.ep_value = [value]
        self.ep_context = [context]

        return experience


    def get_data(self, policy=None):
        """
        Collects single data rollout using specified policy.

        Args:
            policy:     policy to execute

        Returns:
                data rollout dictionary
        """
        if policy is None:
            policy = self.policy

        if not self.terminal_end:
            # Collect rollout:






