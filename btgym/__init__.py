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

from .spaces import DictSpace, ActionDictSpace
from .strategy import BTgymBaseStrategy
from .server import BTgymServer
from .datafeed import BTgymDataset2, BTgymRandomDataDomain, BTgymSequentialDataDomain
from .datafeed import DataSampleConfig, EnvResetConfig
from .dataserver import BTgymDataFeedServer
from .rendering import BTgymRendering
from .envs.base import BTgymEnv
from btgym.envs.multidiscrete import MultiDiscreteEnv
from btgym.envs.portfolio import PortfolioEnv

register(
    id='backtrader-v0000',
    entry_point='btgym.envs:BTgymEnv',
)
