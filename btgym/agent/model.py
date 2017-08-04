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

# import numpy as np
from types import MethodType
import tensorflow as tf



class BTgymModel():
    """
    Base deep Q-model class.
    """
    session = None

    memory_class = None
    memory_params = dict()

    network_class = None
    network_params = dict()

    experience_shape = dict()
    input_shape = None
    valid_actions = []
    scope = 'ddqn_model'

    optimizer_class = tf.train.RMSPropOptimizer
    optimizer_params = dict(
        learning_rate=0.00025,
        decay=0.9,
        momentum=0.0,
        epsilon=1e-10,
    )
    gamma = .97
    tau = 0.001
    t_update_freq = 5

    loss_on_batch = None

    def __init__(self,
                 session,
                 experience_shape,
                 valid_actions,
                 max_episode_length,
                 memory_class,  # storage class
                 network_class,  # network class
                 **kwargs):
        """
        # Arguments:
            session: tf.Session() object
            experience_shape: dict.
            max_episode_length: integer scalar.
            valid_actions: list of actions (integers).
            network: neural network class.
            storage: replay storage class.
            Optional:
                memory_params:
                network_params:
                optimizer_class:
                optimizer_params:
                gamma:
                tau:
                scope: name scope.
        """

        # Update defaults:
        for key, value in kwargs.items():
            if key in self.__dir__():
                setattr(self, key, value)

        self.session = session
        self.experience_shape = experience_shape
        self.state_shape = experience_shape['state_next']
        self.valid_actions = valid_actions
        self.memory_class = memory_class
        self.network_class = network_class

        self.memory_params['experience_shape'] = self.experience_shape
        self.memory_params['max_episode_length'] = max_episode_length
        self.memory_params['session'] = self.session

        with tf.variable_scope(self.scope):  # TODO: move inside specific _logic_constructor()
            # Make storage:
            self.memory = self.memory_class(**self.memory_params)

            # Optimizer:
            self.optimizer = self.optimizer_class(**self.optimizer_params)

            # Define algorithm logic:
            self._logic_constructor()

            # Make network updater:
            self.update_t_network = self._update_estimator_constructor(
                self.q_network,
                self.t_network,
            )
            self.soft_update_t_network = self._soft_update_estimator_constructor(
                self.q_network,
                self.t_network,
                self.tau
            )

    def _soft_update_estimator_constructor(self, estimator1, estimator2, tau):
        """
        Defines copy-work operation graph for double-network algorithms.
        Receives:
            estimator1: instance to update from;
            estimator2: instance to be updated.
            tau: update intensity parameter, <<1.
        Returns:
            instance method:
                _update_estimator(sess):
                    Softly updates model parameters
                    of one estimator towards ones of another.
                    Receives:
                        tf.Session() object.
        """
        e1_params = [t for t in tf.trainable_variables() if estimator1.scope in t.name]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if estimator2.scope in t.name]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self._update_estimator_op = []
        with tf.variable_scope('update_estimator'):
            for e1_p, e2_p in zip(e1_params, e2_params):
                op = e2_p.assign(e1_p.value() * tau + e2_p.value() * (1 - tau))
                self._update_estimator_op.append(op)

        def _update_estimator(self):
            """
            Softly updates model parameters of one estimator towards ones of another.
            sess: tensorflow session instance.
            """
            self.session.run(self._update_estimator_op)

        return MethodType(_update_estimator, self)

    def _update_estimator_constructor(self, estimator1, estimator2):
        """
        Defines copy-work operation graph for double-network algorithms.
        Recieves:
            estimator1: instance to update from;
            estimator2: instance to be updated.
            tau: update intensity parameter, <<1.
        Returns:
            instance method:
                _update_estimator(sess):
                    Softly updates model parameters
                    of one estimator towards ones of another.
                    Receives:
                        tf.Session() object.
        """
        e1_params = [t for t in tf.trainable_variables() if estimator1.scope in t.name]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if estimator2.scope in t.name]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self._update_estimator_op = []
        with tf.variable_scope('update_estimator'):
            for e1_p, e2_p in zip(e1_params, e2_params):
                op = e2_p.assign(e1_p)
                self._update_estimator_op.append(op)

        def _update_estimator(self):
            """
            Softly updates model parameters of one estimator towards ones of another.
            sess: tensorflow session instance.
            """
            self.session.run(self._update_estimator_op)

        return MethodType(_update_estimator, self)

    def _logic_constructor(self):
        """
        OVERRIDE.
        Defines core algorithm logic operations.
        # Sets instance attributes:
            - q_network instance
            - t_network instance
            - tensor holding q-loss estimate across single batch;
            - tensor holding predicted best actions for given batch state input;
            - tensor holding predicted q_values for given batch state input;
            - tensor holding single step algorithm update operation.
        """
        self.input_shape = (None,) + self.state_shape
        self.q_network = None
        self.t_network = None
        self.q_loss_op = None
        self.train_op = None
        self.predict_q_values_op = None
        self.predict_actions_op = None
        raise NotImplementedError

    def predict(self, state):
        """
        [OVERRIDE.]
        Deterministically predicts actions on state batch input.
        # Args:
            state: states batch as [dictionary] of numpy arrays.
        # Returns:
            actions: batch of predicted actions as numpy array.
        """
        return self.session.run(
            self.predict_actions_op,
            feed_dict={
                self.q_network.input: state, # feed values to network input placeholder
            },
        )

    def update(self, global_step):
        """
        OVERRIDE
        Updates model according to algorithm logic.
        """
        raise NotImplementedError







