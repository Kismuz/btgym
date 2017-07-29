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
    session = None

    experience_shape = dict()
    experience = dict()
    init_experience = dict()

    model_class = None

    model_params = dict(
        memory_class=None,
        network_class=None,
        scope='model',
    )

    epsilon_start = 0.99
    epsilon_end = 0.1
    epsilon_decay_steps = 500000

    saver = None
    load_latest_checkpoint = True,
    home_dir = None
    checkpoint_dir = None
    checkpoint_path = None
    monitor_path = None

    scope = 'btgym_base_dqn_agent'

    def __init__(self,
                 session,
                 experience_shape,  # dictionary of external and internal shapes and tf.Dtype's
                 valid_actions,  # list of intergers representing valid actions
                 max_episode_length,  # in number of experiences
                 model_class,  # model to use
                 model_params,  # dict
                 **kwargs):
        """____"""
        self.session = session
        self.experience_shape = experience_shape
        self.experience = self._experience_numpy_constructor(experience_shape)
        self.init_experience = self._experience_numpy_constructor(experience_shape) #copy.deepcopy(self.experience)

        self.max_episode_length = max_episode_length
        self.valid_actions = valid_actions
        self.num_actions = len(self.valid_actions)

        self.model_class = model_class

        self.model_params.update(model_params)
        self.model_params['session'] = self.session
        self.model_params['experience_shape'] = self.experience_shape
        self.model_params['valid_actions'] = self.valid_actions
        self.model_params['max_episode_length'] = self.max_episode_length

        # Update defaults:
        for key, value in kwargs.items():
            if key in self.__dir__():
                setattr(self, key, value)

        with tf.variable_scope(self.scope):
            # Make model:
            self.model = self.model_class(**self.model_params)

            # Make global step variable and methods:
            self.get_global_step, self.global_step_up = self._global_step_constructor()

            # Make e-greedy policy method for choosen estimator:
            self.e_policy = self._epsilon_greedy_policy_constructor()

        # Create linear epsilon decay function of x(step): epsilon =  a * x + b
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

        def _get_global_step(self):
            """
            Receives:
                tf.Session() object.
            Returns:
                current step value.
            """
            return self.session.run(self._global_step_var)

        def _global_step_up(self):
            """
            Increments global step count by 1.
            Receives:
                tf.Session() object.
            Returns:
                New  step value.
            """
            return self.session.run(self._global_step_increment_op)

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

    def _epsilon_greedy_policy_constructor(self):
        """
        Creates an epsilon-greedy policy function.
        Returns:
            instance method:
                _policy_fn(action, epsilon):
                # Args:
                    action: deterministic action.
                    epsilon: e-greedy parameter < 1.
                Returns:
                    probabilities for each valid action in the form of a numpy array of length nA.
        """
        def _policy(self, action, epsilon):
            action_probs = np.ones(self.num_actions, dtype=float) * epsilon / self.num_actions
            action_probs[int(action)] += (1.0 - epsilon)
            return action_probs

        return MethodType(_policy, self)

    def populate_memory(self,sess, env, init_size=None):
        # TODO: REWRITE!
        """
        Populates initial replay memory following e-greedy policy.
        """
        if init_size is None:
            init_size = self.model.memory.replay_memory_init_size
        # How much episodes to add:
        episodes_to_add = int(init_size / self.max_episode_length) - self.model.memory._get_current_size(sess)

        if episodes_to_add > 0:
            for episode in range(episodes_to_add):
                self.experience = copy.deepcopy(self.init_experience)
                self.experience['state_next'] = env.reset()
                self.model.memory.update(sess, self.experience)

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
                    self.model.memory.update(sess, self.experience)
                    # Do NOT increment self.global_step
                    if self.experience['done']:
                        break

    def act(self, state, deterministic=False):
        """
        Emit action on given state.
        # Args:
            state: environment observation state.
            deterministic: [bool] act deterministically or according e-greedy policy.
        # Returns:
            predicted action, integer scalar.
        """
        det_action = self.model.predict(state)

        if deterministic:
            return det_action

        else:
            return np.random.choice(
                self.num_actions,
                p=self.e_policy(
                    det_action,
                    self.epsilon(
                        self.get_global_step()
                    )
                )
            )

    def observe(self, experience):
        """
        Reflect on what we've done.
        """
        # Store experience in replay memory:
        self.model.memory.update(self.session, experience)
        # Perform model update:
        self.model.update(self.get_global_step())
        # Advance global step:
        self.global_step_up()
