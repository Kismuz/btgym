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
import bisect
import datetime
import multiprocessing
import numpy as np
from backtrader.plot import Plot_OldSync


class BTgymPlotter(Plot_OldSync):
    """Hacky way to get cerebro.plot() renderings.
    Overrides default backtrader plotter behaviour.
    """

    def __init__(self, **kwargs):
        """
        pass
        """
        super(BTgymPlotter, self).__init__(**kwargs)

    def savefig(self, fig, filename, width=16, height=9, dpi=300, tight=True,):
        """
        We neither need picture to appear in <stdout> nor file to be written to disk (slow).
        Just set params and return `fig` to be converted to rgb array.
        """
        fig.set_size_inches(width, height)
        fig.set_dpi(dpi)
        fig.set_tight_layout(tight)
        fig.canvas.draw()


class DrawCerebro(multiprocessing.Process):
    """That's the way we plot it...
    """
    def __init__(self, cerebro, width, height, dpi, result_pipe, use=None, rowsmajor=1):
        super(DrawCerebro, self).__init__()
        self.result_pipe = result_pipe
        self.cerebro = cerebro
        self.plotter = BTgymPlotter()
        self.width = width
        self.height = height
        self.dpi = dpi
        self.use = use
        self.rowsmajor=rowsmajor

    def run(self):
        """

        Returns:
             rgb_array.
        """
        fig = self.cerebro.plot(plotter=self.plotter,  # Modified above plotter class, doesnt actually saves anything.
                                savefig=True,
                                width=self.width,
                                height=self.height,
                                dpi=self.dpi,
                                use=self.use,
                                iplot=False,
                                rowsmajor=self.rowsmajor,
                                figfilename='_tmp_btgym_render.png',
                               )[0][0]
        fig.canvas.draw()
        rgb_string = fig.canvas.tostring_rgb()
        rgb_shape = fig.canvas.get_width_height()[::-1] + (3,)
        rgb_array = np.fromstring(rgb_string, dtype=np.uint8, sep='')
        rgb_array = rgb_array.reshape(rgb_shape)

        try:
            self.result_pipe.send(rgb_array)
            self.result_pipe.close()

        except:
            raise RuntimeError('Can not perform episode rendering.\n' +
                               'Hint: check storage consumption or use: render_enabled=False')
        return None

