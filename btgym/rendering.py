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
import logging
import numpy as np
from PIL import Image
import matplotlib

class BTgymRendering():
    """
    Executes rendering of BTgym Environment.
    """
    # Rendering output elements:
    state = None  # featurized state representation to plot,  type=np.array.
    raw_state = None  # raw state as O,H,L,C,V price datalines, type=np.array.
    title = ''  # figure title, type=str.
    box_text = ''  # inline text block, type=str.


    # Plotting controls, can be passed as kwargs:
    params = dict(
        render_type='plot',
        render_figsize=(10, 6),
        render_dpi=300,
        render_plotstyle='seaborn',
        render_cmap='PRGn',
        render_xlabel='Relative timesteps',
        render_ylabel='Value',
        render_title='step: {}, state observation min: {:.4f}, max: {:.4f}',
        render_boxtext=dict(
            fontsize=12,
            fontweight='bold',
            color='w',
            bbox={'facecolor': 'k', 'alpha': 0.3, 'pad': 3},
        ),
        episode_picfilename='btgym_current_episde.png',
        plt_backend='Agg',
    )

    def __init__(self, **kwargs):
        self.log = logging.getLogger('Plotter')
        logging.getLogger().setLevel(logging.WARNING)

        # Update parameters, and set params as instance attributes:
        for key, value in self.params.items():
            if key in kwargs:
                value = kwargs.pop(key)
            setattr(self, key, value)

        # Backend:
        #matplotlib.use(self.plt_backend)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        self.plt = plt
        self.FigureCanvas = FigureCanvas

    def to_string(self, dictionary, excluded=[]):
        """
        Converts given dictionary to more-or-less good looking string.
        """
        text = ''
        for k, v in dictionary.items():
            if k not in excluded:
                if type(v) in [float]:
                    v = '{:.4f}'.format(v)
                text += '{}: {}\n'.format(k, v)
        return text

    def parse_response(self, raw_state, state, reward, info, done,):
        """
        Converts environment response to plotting attributes:
        state, title, text.
        """
        try:
            # State output:
            self.state = np.asarray(state)
            self.raw_state = np.asarray(raw_state)
            assert len(self.state.shape) == 2
            assert len(self.raw_state.shape) == 2

        except:
            raise NotImplementedError('Only 2D observation state rendering supported.')

        # Figure out how to deal with info output:
        try:
            assert type(info[-1]) == dict
            info_dict = info[-1]

        except:
            try:
                assert type(info) == dict
                info_dict = info

            except:
                try:
                    info_dict = {'info': str(dict)}

                except:
                    info_dict = {}

        # Add records:
        info_dict.update(reward=reward, is_done=done)

        # Try to get step information:
        try:
            current_step = info_dict['step']

        except:
            current_step = '--'

        # Compose box text, excl. redundant fields:
        self.box_text = self.to_string(info_dict, excluded=['step'])

        # Make title output:
        self.title = self.render_title.format(current_step, self.state.min(), self.state.max())

    def as_text(self):
        """
        Ascetic: just text output.
        """
        raise NotImplementedError('For python''s sake, install that Matplotlib!')

    def as_image(self): #### STATE REPRESENTATION
        """
        Visualises environment state as image.
        """
        self.plt.figure(figsize=self.render_figsize)
        #self.plt.style.use(self.render_plotstyle)
        self.plt.title(self.title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(self.state.shape[-1] - 1, 0, int(self.state.shape[-1]), dtype=int)
        self.plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in self.plt.xticks()[1][::5]:
            tick.set_visible(True)

        self.plt.xlabel(self.render_xlabel)
        self.plt.ylabel(self.render_ylabel)
        self.plt.grid(False)

        # Add Info box:
        self.plt.text(1, self.state.shape[0]-1, self.box_text, **self.render_boxtext)

        self.plt.imshow(self.state, aspect='auto', cmap=self.render_cmap)
        self.plt.show()

    def episode(self, cerebro):
        """
        Renders entire episode using built-in backtrader plotting feature.
        Returns image as rgb_array.
        """
        try:
            assert cerebro is not None

        except:  # nothing to render
            return (np.random.rand(100,100, 3) * 255).astype(dtype=np.uint8)

        # Save picture to file:
        cerebro.plot(
            savefig=True,
            width=self.render_figsize[0],
            height=self.render_figsize[1],
            dpi=self.render_dpi,
            use=None,
            iplot=False,
            figfilename=self.episode_picfilename,
            **self.params,
        )

        # Reload and convert:
        try:
            episode_rgb_array = np.array(Image.open(self.episode_picfilename))

        except:
            raise FileNotFoundError('Rendered <{}> not found'.format(self.episode_picfilename))

        return [episode_rgb_array]

    def step(self, raw_state, state, reward, done, info, mode=('price')):
        """
        Renders current environment state.
        Returns image as rgb_array.
        """
        rgb_list = []

        self.parse_response(raw_state, state, reward, info, done)

        if 'state' in mode:
            # Render featurized state
            pass

        if 'price' in mode:
            # Render price data
            rgb_list.append(self.draw_plot(raw_state))

        if 'episode' in mode:
            # Load saved file, if any:
            try:
                rgb_list.append(np.array(Image.open(self.episode_picfilename)))

            except:
                self.log.warning('Rendered <{}> not found'.format(self.episode_picfilename))

        return rgb_list

    def draw_plot(self, data):
        """
        Visualises environment state as 2d line plot.
        """
        fig = self.plt.figure(figsize=self.render_figsize, dpi=self.render_dpi,)
        ax = fig.add_subplot(111)

        self.plt.style.use(self.render_plotstyle)
        self.plt.tight_layout(pad=3)
        self.plt.title(self.title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(self.state.shape[-1] - 1, 0, int(self.state.shape[-1]), dtype=int)
        self.plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in self.plt.xticks()[1][::5]:
           tick.set_visible(True)

        self.plt.xlabel(self.render_xlabel)
        self.plt.ylabel(self.render_ylabel)
        self.plt.grid(True)

        # Switch off antialiasing:
        #self.plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
        #self.plt.rcParams['text.antialiased']=False

        # Add Info box:
        self.plt.text(0, data.T.min(), self.box_text, **self.render_boxtext)

        self.plt.plot(data.T)
        fig.canvas.draw()

        # Save it to a numpy array:
        rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        return rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))