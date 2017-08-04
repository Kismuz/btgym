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

from .datafeed import BTgymDataset
from .server import BTgymServer
from .dataserver import BTgymDataFeedServer
from .strategy import BTgymStrategy
from .rendering import BTgymRendering
#from .monitor import BTgymMonitor
from .envs.backtrader import BTgymEnv

register(
    id='backtrader-v0000',
    entry_point='btgym.envs:BTgymEnv',
)
