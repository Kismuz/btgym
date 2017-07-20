###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import os
import numpy as np
import tensorflow as tf
from .memory import BTgymReplayMemory


class BTgymDqnAgent():
    """
    Base Double Q-Network agent with replay memory
    class for episodic tasks with
    bi-modal state space and
    discrete action space.

    Bi-modal observation state shape is defined
    as dictionary of two arbitrary shaped tensors:
    state_shape = dict(external=(N1,N2, ..., Nk),
                       internal=(M1, M2, ..., Ml),)

    Shape of single experience therefore is:
    experience_shape = dict(state_external=state_shape['external'],
                            state_internal=state_shape['internal'],
                            action=(),
                            reward=(),
                            state_internal_next=state_shape['internal'],
                            state_external_next=state_shape['internal'],)
    """
    state_shape = dict(
        external=(None, None),
        internal=(None, None),
    )
    state = dict(
        external=(None, None),
        internal=(None, None),
    )
    # TODO: remove as it is internal replay memory representation:
    experience_shape = dict(
        state_external=state_shape['external'],
        state_internal=state_shape['internal'],
        action=(),
        reward=(),
        state_external_next=state_shape['external'],
        state_internal_next=state_shape['internal'],
    )
    experience = dict(
        state_external=None,
        state_internal=None,
        action=None,
        reward=None,
        state_external_next=None,
        state_internal_next=None,
    )
    action = None
    reward = None

    replay_memory_size = 100000
    replay_memory_init_size = 50000
    epsilon_start = 0.99
    epsilon_end = 0.1
    epsilon_decay_steps = 500000
    gamma = 0.99
    tau = 0.001
    batch_size = 32

    saver = None
    load_latest_checkpoint = True,

    scope = 'btgym_base_q_agent'

    def __init__(self,
                 state_shape,  # dictionary of external and internal shapes
                 valid_actions,  # list of intergers representing valid actions
                 max_episode_length,  # in number of experiences
                 estimator_class,  # class of estimator to use
                 **kwargs):
        """____"""
        self.state_shape = state_shape
        self.estimator_class = estimator_class
        self.max_episode_length = max_episode_length
        self.valid_actions = valid_actions
        self.replay_memory_class = BTgymReplayMemory
        # Update defaults:
        for key, value in kwargs.items():
            if key in self.__dir__():
                setattr(self, key, value)

        with tf.variable_scope(self.scope):
            # Make estimators and updater:
            self.q_estimator = self.estimator_class(
                self.state_shape,
                self.valid_actions,
                **kwargs
            )
            self.t_estimator = self.estimator_class(
                self.state_shape,
                self.valid_actions,
                **kwargs
            )
            self._estimator_update_constructor(
                self.q_estimator,
                self.t_estimator,
            )
            # Make replay memory:
            self.memory = self.replay_memory_class(
                state_shape=self.state_shape,
                max_episode_length=self.max_episode_length,
                max_size=self.replay_memory_size,
                batch_size=self.batch_size,
                scope='replay_memory',
                **kwargs
            )
            # Make global step:
            self._global_step_constructor()

            # Make e-greedy policy function for chosen estimator:
            # TODO: possibly should be part of estimator
            self.policy = self._epsilon_greedy_policy_constructor(self.q_estimator)

            # Call logic graph constructor:
            ##self._logic_constructor(self):


        # Create linear epsilon decay function: epsilon =  a * x + b
        self.epsilon = lambda x: self.epsilon_end if x > self.epsilon_decay_steps else\
            (self.epsilon_end - self.epsilon_start) / self.epsilon_decay_steps * x + self.epsilon_start

        # Define saver et. all:
        if self.saver is not None:
            self.saver = tf.train.Saver()
            # Create directories for checkpoints and summaries:
            self.home_dir = './{}/'.format(self.scope)
            self.checkpoint_dir = os.path.join(self.home_dir, "checkpoints")
            self.checkpoint_path = os.path.join(self.home_dir, "model")
            self.monitor_path = os.path.join(self.home_dir, "monitor")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.monitor_path):
                os.makedirs(self.monitor_path)

    def _global_step_constructor(self):
        """
        Savable global_step constructor.
        """
        self._global_step = tf.Variable(
            0,
            name='global_step',
            trainable=False,
            dtype=tf.int32,
        )
        self._increment_global_step_op = tf.assign_add(self._global_step, 1)

    def get_global_step(self, sess):
        """
        Returns current step value.
        """
        return sess.run(self._global_step)

    def global_step_up(self, sess):
        """
        Increments global step count by 1.
        """
        sess.run(self._increment_global_step_op)

    def save(self, sess):
        """
        Saves current agent state.
        """
        if self.saver is not None:
            with sess.as_default():
                self.saver.save(self.checkpoint_path)

        else:
            raise RuntimeError('Saver for <{}> is not defined.'.format(self.scope))

    def restore(self, sess):
        """
        Restores agent state from latest saved checkpoint if it exists.
        """
        if self.saver is not None:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint and self.load_latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)

        else:
            raise RuntimeError('Saver for <{}> is not defined.'.format(self.scope))

    def _estimator_update_constructor(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: instance to update from;
          estimator2: instance to be updated.
          tau: update intensity parameter, <<1.
        """
        self.e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        self.e1_params = sorted(self.e1_params, key=lambda v: v.name)
        self.e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        self.e2_params = sorted(self.e2_params, key=lambda v: v.name)

        self._update_estimator_op = []
        for e1_p, e2_p in zip(self.e1_params, self.e2_params):
            op = e2_p.assign(e1_p.value() * self.tau + e2_p.value() * (1 - self.tau))
            self._update_estimator_op.append(op)

    def update_estimator(self, sess):
        """
        Softly updates model parameters of one estimator towards ones of another.
        sess: tensorflow session instance.
        """
        sess.run(self._update_estimator_op)

    def _epsilon_greedy_policy_constructor(self, estimator):
        """
        Creates an epsilon-greedy policy based on a given function approximator and epsilon.
        Args:
            estimator: An estimator that returns q values for a given state
            nA: Number of actions in the environment.
        Returns:
            A function that takes the (sess, observation, epsilon) as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """
        nA = len(self.valid_actions)

        def policy_fn(sess, observation, epsilon):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def populate_mempory(self,sess, env):
        """
        Populates initial replay memory following e-greedy policy.
        """
        # Get current memory state (approximately):
        mem_size_steps = self.memory._get_current_size(sess) * self.max_episode_length

        # How much to add:
        need_to_add = self.replay_memory_init_size - mem_size_steps

        if need_to_add > self.max_episode_length:
            state = env.reset()
            for i in range(need_to_add):
                action_probs = self.policy(
                    sess, state, self.epsilon(self.get_global_step(sess))
                )
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                state_next, reward, done, info = env.step(action)
                self.memory.update(state, action, reward, state_next)
                # Do or don't? Rather don't - or else part of epsilons just get waisted:
                #self.global_step_up(sess)
                if done:
                    state = env.reset()

                else:
                    state = state_next

    def _logic_constructor(self):
        """
        Defines agent computational logic.
        """
        raise NotImplementedError

    def act(self, env, action):
        """
        AcT!
        """
        raise NotImplementedError

    def observe(self, env):
        """
        Reflect on what we've done.
        """
        raise NotImplementedError
