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

from backtrader.plot import Plot_OldSync


class BTgymPlotter(Plot_OldSync):
    """
    Hacky way to get cerebro.plot() renderings.
    Overrides default backtrader plotter behaviour.
    """

    def __init__(self):
        """
        pass
        """
        super(BTgymPlotter, self).__init__()

    def savefig(self, fig, filename, width=16, height=9, dpi=300, tight=True):
        """
        We neither need picture to appear in <stdout> nor file to be written (slow).
        Just set params and return `fig` to converted to rgb array.
        """
        fig.set_size_inches(width, height)
        fig.set_dpi(dpi)
        fig.set_tight_layout(tight)
