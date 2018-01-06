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

from gym.envs.registration import register

from .strategy import BTgymBaseStrategy
from .server import BTgymServer
from .datafeed import BTgymDataset, BTgymRandomDataDomain, BTgymSequentialDataDomain
from .dataserver import BTgymDataFeedServer
# from .monitor import BTgymMonitor
from .rendering import BTgymRendering
from .spaces import DictSpace
from .envs.backtrader import BTgymEnv

register(
    id='backtrader-v0000',
    entry_point='btgym.envs:BTgymEnv',
)
