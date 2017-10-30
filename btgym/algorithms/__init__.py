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

from .envs import create_env
from .rollout import Rollout, ExperienceConfig
from .memory import Memory
from .runner import RunnerThread
from .worker import Worker

from .policy import BaseAacAuxPolicy, AacAuxPolicy, BaseAacPolicy, AacPolicy

from .a3c import A3C
from .unreal import Unreal
from .ppo import PPO

from .launcher import Launcher

