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
from types import MethodType
import  itertools
import copy
import numpy as np
import tensorflow as tf
from .memory import BTgymReplayMemory


class BTgymDqnAgent():
    """
    Base deep Q-network agent with replay memory
    class for episodic tasks with nested
    multi-modal state observation and experience shape.

    Experience is unit to store in replay memory,
    defined by `experience_shape` dictionary and
    can be [nested] dictionary of any structure
    with at least these keys presented at top-level:
        `action`,
        `reward`,
        `done`,
        `state_next`;
    every end-level record is tuple describing variable shape and dtype.

    Shape is arbitrary, dtype can be any of valid numpy compatible tf.Dtype's.
    If dtype arg is omitted,
    float32 will be set by default.

    When constructing self.experience variable, serving as buffer when receiving experience from
    environment, tf.Dtype will be substituted with compatible numpy dtype.

    Example:
        robo_experience_shape = dict(
            action=(4,tf.uint8),  # unsigned 8bit integer vector
            reward=(),  # float32 by default, scalar
            done=(tf.bool,),   # boolean, scalar
            state_next=dict(
                internal=dict(
                    hardware=dict(
                        battery=(),  # float32, scalar
                        oil_pressure=(3,),  # got 3 pumps, float32, vector
                        tyre_pressure=(4,),  # for each one, float32, vector
                        checks_passed=(tf.bool,)  # boolean, scalar
                    ),
                    psyche=dict(
                        optimism=(tf.int32,),  # can be high, 32bit int, scalar
                        sincerity=(),  # float32 by default, scalar
                        message=(4,tf.string,),  # knows four phrases
                    )
                ),
                external=dict(
                    camera=(2,180,180,3,tf.uint8),  # binocular rgb 180x180 image, unsigned 8bit integers
                    audio_sensor=(2,320,)  # stereo audio sample buffer, float32
                ),
            ),
            global_training_day=(uint16,)  # just curious how long it took to get to this state.
        )
        Note:
            using of tensorflow tf.Dtypes.
    """
    replay_memory_size = 100000
    replay_memory_init_size = 50000
    batch_size = 32
    epsilon_start = 0.99
    epsilon_end = 0.1
    epsilon_decay_steps = 500000
    gamma = 0.99
    tau = 0.001

    saver = None
    load_latest_checkpoint = True,

    scope = 'btgym_base_q_agent'

    def __init__(self,
                 experience_shape,  # dictionary of external and internal shapes and tf.Dtype's
                 valid_actions,  # list of intergers representing valid actions
                 max_episode_length,  # in number of experiences
                 estimator_class,  # class of estimator to use
                 **kwargs):
        """____"""
        self.experience_shape = experience_shape
        self.experience = self._experience_numpy_constructor(experience_shape)
        self.init_experience = self._experience_numpy_constructor(experience_shape) #copy.deepcopy(self.experience)
        self.estimator_class = estimator_class
        self.max_episode_length = max_episode_length
        self.valid_actions = valid_actions
        self.replay_memory_class = BTgymReplayMemory
        # Update defaults:
        for key, value in kwargs.items():
            if key in self.__dir__():
                setattr(self, key, value)

        with tf.variable_scope(self.scope):
            # Make estimators and update method:
            self.q_estimator = self.estimator_class(
                self.experience_shape,
                self.valid_actions,
                scope='q_estimator',
            )
            self.t_estimator = self.estimator_class(
                self.experience_shape,
                self.valid_actions,
                scope='t_estimator',
            )
            self.update_t_estimator = self._update_estimator_constructor(
                self.q_estimator,
                self.t_estimator,
            )
            # Make replay memory:
            self.memory = self.replay_memory_class(
                experience_shape=self.experience_shape,
                max_episode_length=self.max_episode_length,
                max_size=self.replay_memory_size,
                batch_size=self.batch_size,
                scope='replay_memory',
            )
            # Make global step methods:
            self.get_global_step, self.global_step_up = self._global_step_constructor()

            # Make e-greedy policy method for chosen estimator:
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
            self.home_dir = './{}_home/'.format(self.scope)
            self.checkpoint_dir = os.path.join(self.home_dir, "checkpoints")
            self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
            self.monitor_path = os.path.join(self.home_dir, "monitor")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.monitor_path):
                os.makedirs(self.monitor_path)


    def _experience_numpy_constructor(self, shape_dict):
        """
        Defines experience-holding dictionary according to self.experience_shape.
        Takes:
            nested dictionary of shapes as tuples:
                shape = (dim_0,....,dim_N, [tf.Dtype]).
        Returns:
            nested dictionary of np.dtype arrays.
        Note:
            using of tf.Dtype in experience_shape.
        """
        exp_dict = dict()
        for key, record in shape_dict.items():
            if type(record) == dict:
                exp_dict[key] = self._experience_numpy_constructor(record)
            else:
                # If dtype is not present - set it to np.float32,
                # else - convert from tf.dtype:
                dtype = np.float32
                if len(record) > 0 and type(record[-1]) != int:
                    dtype = record[-1].as_numpy_dtype
                    record = record[0:-1]
                exp_dict[key] = np.zeros(
                    shape=record,
                    dtype=dtype,
                )
        return exp_dict

    def _global_step_constructor(self):
        """
        Savable global_step constructor.
        Returns:
            instance methods:
                _get_global_step(sess);
                    Receives:
                        tf.Session() object.
                    Returns:
                        current step value.
                _global_step_up(sess):
                    Increments global step count by 1.
                    Receives:
                        tf.Session() object.
                    Returns:
                        New step value.
        """
        self._global_step_var = tf.Variable(
            0,
            name='global_step',
            trainable=False,
            dtype=tf.int32,
        )
        self._global_step_increment_op = tf.assign_add(self._global_step_var, 1)

        def _get_global_step(self, sess):
            """
            Receives:
                tf.Session() object.
            Returns:
                current step value.
            """
            return sess.run(self._global_step_var)

        def _global_step_up(self, sess):
            """
            Increments global step count by 1.
            Receives:
                tf.Session() object.
            Returns:
                New  step value.
            """
            return sess.run(self._global_step_increment_op)

        return MethodType(_get_global_step, self), MethodType(_global_step_up, self)

    def save(self, sess):
        """
        Saves current agent state.
        """
        if self.saver is not None:
            self.saver.save(tf.get_default_session(), self.checkpoint_path)

        else:
            raise RuntimeError('Saver for <{}> is not defined.'.format(self.scope))

    def restore(self, sess):
        """
        Restores agent state from latest saved checkpoint if it exists.
        """
        if self.saver is not None:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint and self.load_latest_checkpoint:
                self.saver.restore(sess, latest_checkpoint)

        else:
            raise RuntimeError('Saver for <{}> is not defined.'.format(self.scope))

    def _update_estimator_constructor(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Recieves:
            estimator1: instance to update from;
            estimator2: instance to be updated.
            tau: update intensity parameter, <<1.
        Returns:
            instance method:
                _update_estimator(sess):
                Softly updates model parameters of one estimator towards ones of another.
                Receives:
                    tf.Session() object.
        """
        self.e1_params = [t for t in tf.trainable_variables() if estimator1.scope in t.name]
        self.e1_params = sorted(self.e1_params, key=lambda v: v.name)
        self.e2_params = [t for t in tf.trainable_variables() if estimator2.scope in t.name]
        self.e2_params = sorted(self.e2_params, key=lambda v: v.name)

        self._update_estimator_op = []
        for e1_p, e2_p in zip(self.e1_params, self.e2_params):
            op = e2_p.assign(e1_p.value() * self.tau + e2_p.value() * (1 - self.tau))
            self._update_estimator_op.append(op)

        def _update_estimator(self, sess):
            """
            Softly updates model parameters of one estimator towards ones of another.
            sess: tensorflow session instance.
            """
            sess.run(self._update_estimator_op)

        return MethodType(_update_estimator, self)

    def _epsilon_greedy_policy_constructor(self, estimator):
        """
        Creates an epsilon-greedy policy based on a given function approximator and epsilon.
        Recieves:
            estimator: an estimator instance that returns q values for a given state;
        Returns:
            instance method:
                _policy_fn(sess, observation, epsilon):
                Returns probabilities for each action in the form of a numpy array of length nA.
        """
        nA = len(self.valid_actions)

        def _policy(self, sess, observation, epsilon):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return MethodType(_policy, self)

    def populate_memory(self,sess, env, init_size=None):
        """
        Populates initial replay memory following e-greedy policy.
        """
        if init_size is None:
            init_size = self.replay_memory_init_size
        # How much episodes to add:
        episodes_to_add = int(init_size / self.max_episode_length) - self.memory._get_current_size(sess)

        if episodes_to_add > 0:
            for episode in range(episodes_to_add):
                self.experience = copy.deepcopy(self.init_experience)
                self.experience['state_next'] = env.reset()
                self.memory.update(sess, self.experience)

                for i in itertools.count():
                    action_probs = self.policy(
                        sess,
                        # TODO: fix this!!
                        self.experience['state_next']['external'],
                        self.epsilon(self.get_global_step(sess)),
                    )
                    self.experience['action'] = np.random.choice(
                        np.arange(len(action_probs)),
                        p=action_probs,
                    )
                    (
                        self.experience['state_next'],
                        self.experience['reward'],
                        self.experience['done'],
                        info,
                    ) = env.step(self.experience['action'])
                    self.memory.update(sess, self.experience)
                    # Do NOT increment self.global_step
                    if self.experience['done']:
                        break


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
