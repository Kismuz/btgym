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

# Asynchronous  implementation of several `advantage actor-critic`-style algorithms.
#from .rollout import Rollout, make_data_getter
#from .memory import Memory
from btgym.algorithms.runner.threadrunner import RunnerThread
from .aac import BaseAAC, Unreal, A3C, PPO
from .envs import AtariRescale42x42
from .launcher import Launcher
from .policy import BaseAacPolicy, Aac1dPolicy, StackedLstmPolicy, AacStackedRL2Policy
from .worker import Worker

