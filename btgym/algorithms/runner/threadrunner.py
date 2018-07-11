# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#

from logbook import Logger, StreamHandler, WARNING
import sys

import six.moves.queue as queue
import threading

from btgym.algorithms.runner import BaseEnvRunnerFn


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
                 runner_fn_ref=BaseEnvRunnerFn,
                 memory_config=None,
                 log_level=WARNING,
                 **kwargs):
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
            runner_fn_ref:          callable defining runner execution logic
            memory_config:          replay memory configuration dictionary
            log_level:              int, logbook.level
        """
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.rollout_length = rollout_length
        self.env = env
        self.last_features = None
        self.policy = policy
        self.runner_fn_ref = runner_fn_ref
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.episode_summary_freq = episode_summary_freq
        self.env_render_freq = env_render_freq
        self.task = task
        self.test = test
        self.ep_summary = ep_summary
        self.memory_config = memory_config
        self.log_level = log_level
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('ThreadRunner_{}'.format(self.task), level=self.log_level)

    def start_runner(self, sess, summary_writer, **kwargs):
        try:
            self.sess = sess
            self.summary_writer = summary_writer
            self.start()

        except:
            msg = 'start() exception occurred.\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError

    def run(self):
        """Just keep running."""
        try:
            with self.sess.as_default():
                self._run()

        except:
            msg = 'RunTime exception occurred.\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError

    def _run(self):
        rollout_provider = self.runner_fn_ref(
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
            self.memory_config,
            self.log
        )
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

