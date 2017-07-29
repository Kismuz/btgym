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
import tensorflow as tf


class BTgymNeuralNetwork():
    """
    Neural network base class wrapper for BTgymModel
    """
    input = None
    output = None

    def __init__(self, input_ts, valid_actions, scope='n_network'):
        """
        Creates neural network with inputs and outputs accepted by BTgymModel class
        # Sets
            self.input: input as tensor,enables input placeholder bypassing.
            self.output: dictionary of output tensors.
        # Arguments
             input_pl: input placeholder or tensor.
             scope: name scope.
        """
        self.valid_actions = valid_actions
        self.scope = scope
        with tf.variable_scope(scope):
            self.input = tf.identity(
                input_ts,
                name='input_tensor'
            )
            self.output = self._network_constructor()

    def _network_constructor(self):
        """
        Override.
        Defines operations for neural network.
        # Returns:
            dictionary of tensors, holding network output[s].
        """
        raise NotImplementedError
