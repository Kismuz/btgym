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


class BTgymAgent():
    """
    Agent base class for episodic tasks with
    bi-modal state space,
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
    scope = 'btgym_base_agent'
    state_shape = dict(
        external=(None, None),
        internal=(None, None),
    )
    experience_shape = dict(
        state_external=state_shape['external'],
        state_internal=state_shape['internal'],
        action=(),
        reward=(),
        state_external_next=state_shape['external'],
        state_internal_next=state_shape['internal'],
    )
    state = dict()
    action = None
    reward = None
    experience = dict(
        state_external=None,
        state_internal=None,
        action=None,
        reward=(),
        state_external_next=None,
        state_internal_next=None,
    )

    valid_actions = []

    estimator1 = None
    estimator2 = None
    logic = None
    memory = None

    saver = None
    home_dir = None
    load_latest_checkpoint = True



    def __init__(self):
        self.logic
        self.estimator
        self.memory