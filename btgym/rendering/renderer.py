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
from logbook import Logger, StreamHandler, WARNING
import sys
import numpy as np

#from .plotter import BTgymPlotter
from .plotter import DrawCerebro

class BTgymRendering():
    """
    Handles BTgym Environment rendering.

    Note:
        Call `initialize_pyplot()` method before first render() call!
    """
    # Here we'll keep last rendered image for each rendering mode:
    rgb_dict = dict()
    render_modes = ['episode', 'human']
    params = dict(
        # Plotting controls, can be passed as kwargs:
        render_state_as_image=True,
        render_state_channel=0,
        render_size_human=(6, 3.5),
        render_size_state=(7, 3.5),
        render_size_episode=(12,8),
        render_rowsmajor_episode=1,
        render_dpi=75,
        render_plotstyle='seaborn',
        render_cmap='PRGn',
        render_xlabel='Relative timesteps',
        render_ylabel='Value',
        render_title='local step: {}, state observation min: {:.4f}, max: {:.4f}',
        render_boxtext=dict(fontsize=12,
                            fontweight='bold',
                            color='w',
                            bbox={'facecolor': 'k', 'alpha': 0.3, 'pad': 3},
                            ),
        plt_backend='Agg',  # Not used.
    )
    enabled = True
    ready = False

    def __init__(self, render_modes, **kwargs):
        """
        Plotting controls, can be passed as kwargs.

        Args:
            render_state_as_image=True,
            render_state_channel=0,
            render_size_human=(6, 3.5),
            render_size_state=(7, 3.5),
            render_size_episode=(12,8),
            render_dpi=75,
            render_plotstyle='seaborn',
            render_cmap='PRGn',
            render_xlabel='Relative timesteps',
            render_ylabel='Value',
            render_title='local step: {}, state observation min: {:.4f}, max: {:.4f}',
            render_boxtext=dict(fontsize=12,
                                fontweight='bold',
                                color='w',
                                bbox={'facecolor': 'k', 'alpha': 0.3, 'pad': 3},
                                )
        """
        # Update parameters with relevant kwargs:
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key] = value

        # Unpack it as attributes:
        for key, value in self.params.items():
                setattr(self, key, value)

        # Logging:
        if 'log_level' not in dir(self):
            self.log_level = WARNING

        StreamHandler(sys.stdout).push_application()
        self.log = Logger('BTgymRenderer', level=self.log_level)

        #from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        #self.FigureCanvas = FigureCanvas

        self.plt = None  # Will set it inside server process when calling initialize_pyplot().

        #self.plotter = BTgymPlotter() # Modified bt.Cerebro() plotter, to get episode renderings.

        # Set empty plugs for each render mode:
        self.render_modes = render_modes
        for mode in self.render_modes:
            self.rgb_dict[mode] = self.rgb_empty()

    def initialize_pyplot(self):
        """
        Call me before use!
        [Supposed to be done inside already running server process]
        """
        if not self.ready:
            from multiprocessing import Pipe
            self.out_pipe, self.in_pipe = Pipe()

            if self.plt is None:
                import matplotlib
                matplotlib.use(self.plt_backend, force=True)
                import matplotlib.pyplot as plt

            self.plt = plt
            self.ready = True

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

    def parse_response(self, state, mode, reward, info, done,):
        """
        Converts environment response to plotting attributes:
        state, title, text.
        """
        if len(state[mode].shape) <= 2:
            state = np.asarray(state[mode])

        elif len(state[mode].shape) == 3:
            if state[mode].shape[1] == 1:
                # Assume 2nd dim (H) is fake expansion for 1D input, so can render all channels:
                state = np.asarray(state[mode][:, 0, :])

            else:
                # Assume it is HWC 2D input, only can render single channel:
                state = np.asarray(state[mode][:, :, self.render_state_channel])

        else:
            raise NotImplementedError(
                '2D rendering can be done for obs. state tensor with rank <= 3; ' +\
                'got state shape: {}'.format(np.asarray(state[mode]).shape))

        # Figure out how to deal with info output:
        try:
            assert type(info[-1]) == dict
            info_dict = info[-1]

        except AssertionError:
            try:
                assert type(info) == dict
                info_dict = info

            except AssertionError:
                try:
                    info_dict = {'info': str(dict)}

                except:
                    info_dict = {}

        # Add records:
        info_dict.update(reward=reward, is_done=done,)

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

    def render(self, mode_list, cerebro=None, step_to_render=None, send_img=True):
        """
        Renders given mode if possible, else
        just passes last already rendered image.
        Returns rgb image as numpy array.

        Logic:
            - If `cerebro` arg is received:
                render entire episode, using built-in backtrader plotting feature,
                update stored `episode` image.

            - If `step_to_render' arg is received:
                - if mode = 'raw_state':
                    render current state observation in conventional 'price' format,
                    update stored `raw_state` image;
                - if mode = something_else':
                    visualise observation as 'seen' by agent,
                    update stored 'agent' image.

        Returns:
             `mode` image.

        Note:
            It can actually return several modes in a single dict.
            It prevented by Gym modes convention, but done internally at the end of the episode.
        """
        if type(mode_list) == str:
            mode_list = [mode_list]

        if cerebro is not None:
            self.rgb_dict['episode'] = self.draw_episode(cerebro)
            self.log.debug('Episode rendering done.')
            # Try to render given episode:
            #try:
                # Get picture of entire episode:
            #fig = cerebro.plot(plotter=self.plotter,  # Modified plotter class, doesnt actually save anything.
            #                   savefig=True,
            #                   width=self.render_size_episode[0],
            #                   height=self.render_size_episode[1],
            #                   dpi=self.render_dpi,
            #                   use=None, #self.plt_backend,
            #                   iplot=False,
            #                   figfilename='_tmp_btgym_render.png',
            #                   )[0][0]

            #fig.canvas.draw()
            #rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            #self.rgb_dict['episode'] = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # Clean up:
            #self.plt.gcf().clear()
            #self.plt.close(fig)

            #except:
                # Just keep previous rendering
             #   pass

        if step_to_render is not None:
            # Perform step rendering:

            # Unpack:
            raw_state, state, reward, done, info = step_to_render

            for mode in mode_list:
                if mode in self.render_modes and mode not in ['episode', 'human']:
                    # Render user-defined (former agent) mode state:
                    agent_state, title, box_text = self.parse_response(state, mode, reward, info, done)
                    if self.render_state_as_image:
                        self.rgb_dict[mode] = self.draw_image(agent_state,
                                                                 figsize=self.render_size_state,
                                                                 title='{} / {}'.format(mode, title),
                                                                 box_text=box_text,
                                                                 ylabel=self.render_ylabel,
                                                                 xlabel=self.render_xlabel,
                                                                 )
                    else:
                        self.rgb_dict[mode] = self.draw_plot(agent_state,
                                                                figsize=self.render_size_state,
                                                                title='{} / {}'.format(mode, title),
                                                                box_text=box_text,
                                                                ylabel=self.render_ylabel,
                                                                xlabel=self.render_xlabel,
                                                                )

                if 'human' in mode:
                    # Render `human` state:
                    human_state, title, box_text = self.parse_response(raw_state, mode, reward, info, done)
                    self.rgb_dict['human'] = self.draw_plot(human_state,
                                                            figsize=self.render_size_human,
                                                            title=title,
                                                            box_text=box_text,
                                                            ylabel='Price',
                                                            xlabel=self.render_xlabel,
                                                            line_labels=['Open', 'High', 'Low', 'Close'],
                                                            )
            if send_img:
                return self.rgb_dict

        else:
            # this case is for internal use only;
            # now `mode` supposed to contain several modes, let's return dictionary of arrays:
            return_dict = dict()
            for entry in mode_list:
                if entry in self.rgb_dict.keys():
                    # ...and it is legal:
                    return_dict[entry] = self.rgb_dict[entry]

                else:
                    return_dict[entry] = self.rgb_empty()

            return return_dict

    def draw_plot(self, data, figsize=(10,6), title='', box_text='', xlabel='X', ylabel='Y', line_labels=None):
        """
        Visualises environment state as 2d line plot.
        Retrurns image as rgb_array.

        Args:
            data:           np.array of shape [num_values, num_lines]
            figsize:        figure size (in.)
            title:
            box_text:
            xlabel:
            ylabel:
            line_labels:    iterable holding line legends as str

        Returns:
                rgb image as np.array of size [with, height, 3]
        """
        if line_labels is None:
            # If got no labels - make it numbers:
            if len(data.shape) > 1:
                line_labels = ['line_{}'.format(i) for i in range(data.shape[-1])]
            else:
                line_labels = ['line_0']
                data = data[:, None]
        else:
            assert len(line_labels) == data.shape[-1], \
                'Expected `line_labels` kwarg consist of {} names, got: {}'. format(data.shape[-1], line_labels)

        fig = self.plt.figure(figsize=figsize, dpi=self.render_dpi, )
        #ax = fig.add_subplot(111)

        self.plt.style.use(self.render_plotstyle)
        self.plt.title(title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(data.shape[0] - 1, 0, int(data.shape[0]), dtype=int)
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
        self.plt.text(0, data.min(), box_text, **self.render_boxtext)

        for line, label in enumerate(line_labels):
            self.plt.plot(data[:, line], label=label)
        self.plt.legend()
        self.plt.tight_layout()

        fig.canvas.draw()

        # Save it to a numpy array:
        rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        # Clean up:
        self.plt.close(fig)
        #self.plt.gcf().clear()

        return rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def draw_image(self, data, figsize=(12,6), title='', box_text='', xlabel='X', ylabel='Y', line_labels=None):
        """
        Visualises environment state as image.
        Returns rgb_array.
        """
        fig = self.plt.figure(figsize=figsize, dpi=self.render_dpi, )
        #ax = fig.add_subplot(111)

        self.plt.style.use(self.render_plotstyle)
        self.plt.title(title)

        # Plot x axis as reversed time-step embedding:
        xticks = np.linspace(data.shape[0] - 1, 0, int(data.shape[0]), dtype=int)
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
        #self.log.warning('render_data_shape:{}'.format(data.shape))

        # Add Info box:
        self.plt.text(0, data.shape[1] - 1, box_text, **self.render_boxtext)

        im = self.plt.imshow(data.T, aspect='auto', cmap=self.render_cmap)
        self.plt.colorbar(im, use_gridspec=True)

        self.plt.tight_layout()

        fig.canvas.draw()

        # Save it to a numpy array:
        rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        # Clean up:
        self.plt.close(fig)
        #self.plt.gcf().clear()

        #ax.cla()
        return rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def draw_episode(self, cerebro):
        """
        Hacky way to render episode.
        Due to backtrader/matplotlib memory leaks have to encapsulate it in separate process.
        Strange but reliable. PID's are driving crazy.

        Args:
            cerebro instance

        Returns:
            rgb array.
        """
        draw_process = DrawCerebro(cerebro=cerebro,
                                   width=self.render_size_episode[0],
                                   height=self.render_size_episode[1],
                                   dpi=self.render_dpi,
                                   result_pipe=self.in_pipe,
                                   rowsmajor=self.render_rowsmajor_episode,
                                   )

        draw_process.start()
        #print('Plotter PID: {}'.format(draw_process.pid))
        try:
            rgb_array = self.out_pipe.recv()

            draw_process.terminate()
            draw_process.join()

            return rgb_array

        except:
            return self.rgb_empty()


class BTgymNullRendering():
    """
    Empty renderer to use when resources are concern.
    """
    enabled = False
    def __init__(self, *args, **kwargs):
        self.plug = (np.random.rand(100, 200, 3) * 255).astype(dtype=np.uint8)
        self.params = {'rendering': 'disabled'}
        self.render_modes = []
        # self.log_level = WARNING
        # StreamHandler(sys.stdout).push_application()
        # self.log = Logger('BTgymRenderer', level=self.log_level)

    def initialize_pyplot(self):
        pass

    def render(self, mode_list, **kwargs):
        # self.log.debug('render() call to environment with disabled rendering. Returning dict of null-images.')
        if type(mode_list) == str:
            mode_list = [mode_list]
        rgb_dict = {}
        for mode in mode_list:
            rgb_dict[mode] = self.plug

        return rgb_dict

    def draw_plot(self, *args, **kwargs):
        # self.log.debug('draw_plot() call to environment with disabled rendering. Returning null-image.')
        return self.plug

    def draw_image(self, *args, **kwargs):
        # self.log.debug('draw_image() call to environment with disabled rendering. Returning null-image.')
        return self.plug

    def draw_episode(self, *args, **kwargs):
        # self.log.debug('draw_episode() call to environment with disabled rendering. Returning null-image.')
        return self.plug
