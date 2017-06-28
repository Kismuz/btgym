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

class BTgymRendering():
    """
    Executes BTgym Environment rendering.
    """
    # Here we'll keep last rendered image for each rendering mode:
    rgb_dict = dict()

    # Plotting controls, can be passed as kwargs:
    render_agent_as_image = True
    render_size_human = (10, 4)
    render_size_agent = (10, 4)
    render_size_episode = (15,10)
    render_dpi=75
    render_plotstyle = 'seaborn'
    render_cmap = 'PRGn'
    render_xlabel = 'Relative timesteps'
    render_ylabel = 'Value'
    render_title = 'step: {}, state observation min: {:.4f}, max: {:.4f}'
    render_boxtext = dict(fontsize=12,
                          fontweight='bold',
                          color='w',
                          bbox={'facecolor': 'k', 'alpha': 0.3, 'pad': 3},
                          )
    plt_backend = 'Agg'

    def __init__(self, render_modes, **kwargs):
        """___"""
        # Update parameters with kwargs:
        self.kwargs = kwargs
        for key, value in self.kwargs.items():
            if key in dir(self):
                setattr(self, key, value)

        # To log or not:
        if 'log' not in dir(self):
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

        #self.plotter = BTgymCerebroPlotter()
        #from backtrader.plot import Plot_OldSync as Plotter
        # Backend:
        #matplotlib.use(self.plt_backend)
        import matplotlib.pyplot as plt
        #from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        #self.FigureCanvas = FigureCanvas

        self.plt = plt
        #self.plt.ioff()
        self.plt.style.use(self.render_plotstyle)

        # initially plug entries for each render mode:
        for mode in render_modes:
            self.rgb_dict[mode] = self.rgb_empty()

        #self.plt.ion()

    def to_string(self, dictionary, excluded=[]):
        """
        Converts given dictionary to more-or-less good looking `text block` string.
        """
        text = ''
        for k, v in dictionary.items():
            if k not in excluded:
                if type(v) in [float]:
                    v = '{:.4f}'.format(v)
                text += '{}: {}\n'.format(k, v)
        return text[:-1]

    def rgb_empty(self):
        """
        Returns empty 'plug' image.
        """
        return (np.random.rand(100, 200, 3) * 255).astype(dtype=np.uint8)

    def parse_response(self, state, reward, info, done,):
        """
        Converts environment response to plotting attributes:
        state, title, text.
        """
        try:
            # State output:
            state = np.asarray(state)
            assert len(state.shape) == 2

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

        # Set box text, excluding redundant fields:
        box_text = self.to_string(info_dict, excluded=['step'])

        # Set title output:
        title = self.render_title.format(current_step, state.min(), state.max())

        return state, title, box_text

    def render(self, mode, cerebro=None, step_to_render=None, ):
        """
        Renders entire episode using built-in backtrader plotting feature,
        or just passes last already rendered step.
        Returns dict with image as rgb_array.
        If cerebro arg is received - renders entire episode
        using built-in backtrader plotting feature and stores to rgb_dict.
        If 'step' arg is received - renders it according 'mode' recieved.
        Returns rgb_dict entry with 'mode' arg key.
        """

        if cerebro is not None:
            # Try to render given episode:
            try:
                # Get picture of entire episode:
                fig = cerebro.plot(#plotter=self.plotter,  # Modified plotter class, doesnt actually save anything.
                                   savefig=True,
                                   width=self.render_size_episode[0],
                                   height=self.render_size_episode[1],
                                   dpi=self.render_dpi,
                                   use=self.plt_backend,
                                   iplot=False,
                                   figfilename='_tmp_btgym_render.png',
                                   **self.kwargs,
                                   )[0][0]

                fig.canvas.draw()

                rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

                self.rgb_dict['episode'] = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            except:
                # Just keep previous rendering
                pass

        if step_to_render is not None:
            # Perform step rendering:

            # Unpack:
            raw_state, state, reward, done, info = step_to_render

            # Render `agent` state:
            agent_state, title, box_text = self.parse_response(state, reward, info, done)
            if self.render_agent_as_image:
                self.rgb_dict['agent'] = self.draw_image(agent_state,
                                                         figsize=self.render_size_agent,
                                                         title=title,
                                                         box_text=box_text,
                                                         ylabel=self.render_ylabel,
                                                         xlabel=self.render_xlabel,
                                                         )
            else:
                self.rgb_dict['agent'] = self.draw_plot(agent_state,
                                                        figsize=self.render_size_agent,
                                                        title=title,
                                                        box_text=box_text,
                                                        ylabel=self.render_ylabel,
                                                        xlabel=self.render_xlabel,
                                                        )

            # Render `human` state:
            human_state, title, box_text = self.parse_response(raw_state, reward, info, done)
            self.rgb_dict['human'] = self.draw_plot(human_state,
                                                    figsize=self.render_size_human,
                                                    title=title,
                                                    box_text=box_text,
                                                    ylabel='Price',
                                                    xlabel=self.render_xlabel,
                                                    )

        # Now return what requested by key image:
        if mode in self.rgb_dict.keys():
            return self.rgb_dict[mode]

        else:
            return self.rgb_empty()

    def draw_plot(self, data, figsize=(10,6), title='', box_text='', xlabel='X', ylabel='Y'):
        """
        Visualises environment state as 2d line plot.
        Retrurns image as rgb_array.
        """
        fig = self.plt.figure(figsize=figsize, dpi=self.render_dpi, )
        #ax = fig.add_subplot(111)

        self.plt.style.use(self.render_plotstyle)
        self.plt.title(title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(data.shape[-1] - 1, 0, int(data.shape[-1]), dtype=int)
        self.plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in self.plt.xticks()[1][::5]:
            tick.set_visible(True)

        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.grid(True)

        # Switch off antialiasing:
        #self.plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
        #self.plt.rcParams['text.antialiased']=False

        # Add Info box:
        self.plt.text(0, data.T.min(), box_text, **self.render_boxtext)

        self.plt.plot(data.T)
        self.plt.tight_layout()

        fig.canvas.draw()

        # Save it to a numpy array:
        rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        return rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def draw_image(self, data, figsize=(12,6), title='', box_text='', xlabel='X', ylabel='Y'):
        """
        Visualises environment state as image.
        Returns rgb_array.
        """
        fig = self.plt.figure(figsize=figsize, dpi=self.render_dpi, )
        #ax = fig.add_subplot(111)

        self.plt.style.use(self.render_plotstyle)
        self.plt.title(title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(data.shape[-1] - 1, 0, int(data.shape[-1]), dtype=int)
        self.plt.xticks(xticks.tolist(), (- xticks[::-1]).tolist(), visible=False)

        # Set every 5th tick label visible:
        for tick in self.plt.xticks()[1][::5]:
            tick.set_visible(True)

        #self.plt.yticks(visible=False)

        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.grid(False)

        # Switch off antialiasing:
        # self.plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
        # self.plt.rcParams['text.antialiased']=False

        # Add Info box:
        self.plt.text(0, data.shape[0] - 1, box_text, **self.render_boxtext)

        im = self.plt.imshow(data, aspect='auto', cmap=self.render_cmap)
        self.plt.colorbar(im, use_gridspec=True)

        self.plt.tight_layout()

        fig.canvas.draw()

        # Save it to a numpy array:
        rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        return rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
