###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin, muzikinae@gmail.com
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
#logging.basicConfig(format='%(name)s: %(message)s')
import time
import zmq
import os

import gym
from gym import error, spaces
#from gym import utils
#from gym.utils import seeding, closer

import backtrader as bt

from btgym import BTgymServer, BTgymStrategy, BTgymDataset, BTgymRendering

from btgym.rendering import BTgymNullRendering

############################## OpenAI Gym Environment  ##############################


class BTgymEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for Backtrader backtesting/trading library.
    """
    metadata = {'render.modes': ['human', 'agent', 'episode',]}
    # `episode` - plotted episode results.
    # `human` - state observation in conventional human-readable format.
    # `agent` - state observation as seen by agent.

    # Dataset:
    dataset = None  # BTgymDataset instance.

    # Backtrader engine:
    engine = None  # bt.Cerbro subclass for server to execute.

    # Strategy:
    strategy = None  # strategy to use if no <engine> class been passed.

    # Server and network:
    server = None  # Server process.
    context = None  # ZMQ context.
    socket = None  # ZMQ socket, client side.
    port = 5500  # network port to use.
    network_address = 'tcp://127.0.0.1:'  # using localhost.
    ctrl_actions = ('_done', '_reset', '_stop', '_getstat', '_render')  # server control messages.

    # Rendering:
    render_enabled = True
    renderer = None  # Rendering support.
    rendered_rgb = dict()  # Keep last rendered images for each mode.

    # Logging:
    log = None
    verbose = 0  # verbosity mode: 0 - silent, 1 - info, 2 - debugging level (lot of traffic!).

    def __init__(self, *args, **kwargs):
        """
        Environment kwargs applying logic:

        if <engine> kwarg is given:
            do not use default engine and strategy parameters;
            ignore <strategy> kwarg and all strategy and engine-related kwargs.

        else (no <engine>):
            use default engine parameters;
            if any engine-related kwarg is given:
                override corresponding default parameter;

            if <strategy> is given:
                do not use default strategy parameters;
                if any strategy related kwarg is given:
                    override corresponding strategy parameter;

            else (no <strategy>):
                use default strategy parameters;
                if any strategy related kwarg is given:
                    override corresponding strategy parameter;

        if <dataset> kwarg is given:
            do not use default dataset parameters;
            ignore dataset related kwargs;

        else (no <dataset>):
            use default dataset parameters;
                if  any dataset related kwarg is given:
                    override corresponding dataset parameter;

        If any <other> kwarg is given:
            override corr. default parameter.
        """

        # Parameters and default values:
        self.params = dict(

            # Backtrader engine mandatory parameters:
            engine=dict(
                start_cash=10.0,  # initial trading capital.
                broker_commission=0.001,  # trade execution commission, default is 0.1% of operation value.
                fixed_stake=10,  # single trade stake is fixed type by def.
            ),
            # Dataset mandatory parameters:
            dataset = dict(
                filename=None,
            ),
            strategy = dict(),
            render = dict(),
        )
        p2 = dict(
            # Strategy related parameters:
            state_shape=None,
                # observation state shape, by convention last dimension is time embedding;
                # one can define any shape; match env.observation_space.shape.
            state_low=None,  # observation space state min/max values,
            state_high=None,  # if set to None - absolute min/max values from BTgymDataset will be used.
            drawdown_call=None,  # episode maximum drawdown threshold, default is 90% of initial value.
            portfolio_actions=None,
                # agent actions,
                # should consist with BTgymStrategy order execution logic;
                # defaults are: 0 - 'do nothing', 1 - 'buy', 2 - 'sell', 3 - 'close position'.
            skip_frame=None,
                # Number of environment steps to skip before returning next response,
                # e.g. if set to 10 -- agent will interact with environment every 10th episode step;
                # Every other step agent's action is assumed to be 'hold'.
                # Note: INFO part of environment response is a list of all skipped frame's info's,
                #       i.e. [info[-9], info[-8], ..., info[0].
        )

        # Update self attributes, remove used kwargs:
        for key in dir(self):
            if key in kwargs.keys():
                setattr(self, key, kwargs.pop(key))

        # Verbosity control:
        self.log = logging.getLogger('Env')

        log_levels = [(0, 'WARNING'), (1, 'INFO'), (2, 'DEBUG'),]

        for key, level in log_levels:
            if key == self.verbose:
                self.log.setLevel(level)

        # Network parameters:
        self.network_address += str(self.port)

        # Set server rendering:
        if self.render_enabled:
            self.renderer = BTgymRendering(self.metadata['render.modes'], **kwargs)

        else:
            self.renderer = BTgymNullRendering()
            self.log.info('Rendering disabled. Call to render() will return null-plug image.')

        # Append logging:
        self.renderer.log = self.log

        # Update params -1: pull from renderer, remove used kwargs:
        self.params['render'].update(self.renderer.params)
        for key in self.params['render'].keys():
            if key in kwargs.keys():
                _ = kwargs.pop(key)

        # DATASET preparation:
        #
        if self.dataset is not None:
            # If BTgymDataset instance has been passed:
            # do nothing.
            msg = 'Custom Dataset class used.'

        else:
            # If no BTgymDataset has been passed,
            # Make default dataset with given CSV file:
            try:
                os.path.isfile(str(self.params['dataset']['filename']))

            except:
                raise FileNotFoundError('Dataset source data file not specified/not found')

            # Use kwargs to instantiate dataset:
            self.dataset = BTgymDataset(**kwargs)
            msg = 'Base Dataset class used.'

        # Append logging:
        self.dataset.log = self.log

        # Update params -2: pull from dataset, remove used kwargs:
        self.params['dataset'].update(self.dataset.params)
        for key in self.params['dataset'].keys():
            if key in kwargs.keys():
                _ = kwargs.pop(key)

        self.log.info(msg)

        # ENGINE preparation:

        # Update params -3: pull engine-related kwargs, remove used:
        for key in self.params['engine'].keys():
            if key in kwargs.keys():
                self.params['engine'][key] = kwargs.pop(key)

        if self.engine is not None:
            # If full-blown bt.Cerebro() subclass has been passed:
            # Update info:
            msg = 'Custom Cerebro class used.'
            self.strategy = msg
            for key in self.params['engine'].keys():
                self.params['engine'][key] = msg

        # Note: either way, bt.observers.DrawDown observer [and logger] will be added to any BTgymStrategy instance
        # by BTgymServer process at runtime.

        else:
            # Default configuration for Backtrader computational engine (Cerebro),
            # if no bt.Cerebro() custom subclass has been passed,
            # get base class Cerebro(), using kwargs on top of defaults:
            self.engine = bt.Cerebro()
            msg = 'Base Cerebro class used.'

            # First, set STRATEGY configuration:
            if self.strategy is not None:
                # If custom strategy has been passed:
                msg2 = 'Custom Strategy class used.'

            else:
                # Base class strategy :
                self.strategy = BTgymStrategy
                msg2 = 'Base Strategy class used.'

            # Add, using kwargs on top of defaults:
            strat_idx = self.engine.addstrategy(self.strategy, **kwargs)

            msg += ' ' + msg2

            # Second, set Cerebro-level configuration:
            self.engine.broker.setcash(self.params['engine']['start_cash'])
            self.engine.broker.setcommission(self.params['engine']['broker_commission'])
            self.engine.addsizer(bt.sizers.SizerFix, stake=self.params['engine']['fixed_stake'])

        self.log.info(msg)

        # Define observation space shape, minimum / maximum values and agent action space.
        # Retrieve values from configured engine or...

        # ...Update params -4:
        # Pull strategy defaults to environment params dict :
        for t_key, t_value in self.engine.strats[0][0][0].params._gettuple():
            self.params['strategy'][t_key] =  t_value

        # Update it with values from strategy 'passed-to params':
        for key, value in self.engine.strats[0][0][2].items():
            self.params['strategy'][key] = value

        # For min/max, if not been set explicitly,
        # the only sensible way is to infer from raw Dataset price values:
        if self.params['strategy']['state_low'] is None or self.params['strategy']['state_high'] is None:

            # Get dataset statistic:
            self.dataset_stat = self.dataset.describe()

            # Exclude 'volume' from columns we count:
            data_columns = list(self.dataset.names)
            data_columns.remove('volume')

            # Override with absolute price min and max values:
            self.params['strategy']['state_low'] =\
            self.engine.strats[0][0][2]['state_low'] =\
                self.dataset_stat.loc['min', data_columns].min()

            self.params['strategy']['state_high'] =\
            self.engine.strats[0][0][2]['state_high'] =\
                self.dataset_stat.loc['max', data_columns].max()

            self.log.info('Inferring obs. space high/low form dataset: {:.6f} / {:.6f}.'.
                          format(self.params['strategy']['state_low'] , self.params['strategy']['state_high']))

        # Set observation space shape from engine/strategy parameters:
        self.observation_space = spaces.Box(low=self.params['strategy']['state_low'],
                                            high=self.params['strategy']['state_high'],
                                            shape=self.params['strategy']['state_shape'],
                                            )

        self.log.debug('Obs. shape: {}'.format(self.observation_space.shape))
        self.log.debug('Obs. min:\n{}\nmax:\n{}'.format(self.observation_space.low, self.observation_space.high))

        # Set action space and corresponding server messages:
        self.action_space = spaces.Discrete(len(self.params['strategy']['portfolio_actions']))
        self.server_actions = self.params['strategy']['portfolio_actions']

        # Finally:
        self.server_response = None
        self.env_response = None

        # self._start_server()  # ... not shure

        self.log.info('Environment is ready.')

    def _start_server(self):
        """
        Configures backtrader REQ/REP server instance and starts server process.
        """
        # Ensure network resources:
        # 1. Release client-side, if any:
        if self.context:
            self.context.destroy()
            self.socket = None

        # 2. Kill any process using server port:
        cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.port)
        os.system(cmd)

        # Set up client channel:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.network_address)

        # Configure and start server:
        self.server = BTgymServer(dataset=self.dataset,
                                  cerebro=self.engine,
                                  render=self.renderer,
                                  network_address=self.network_address,
                                  log=self.log)
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup
        time.sleep(1)

        self.log.info('Server started, pinging {} ...'.format(self.network_address))
        self.socket.send_pyobj({'ctrl': 'ping!'})
        self.server_response = self.socket.recv_pyobj()
        self.log.info('Server seems ready with response: <{}>'.format(self.server_response))

        self._closed = False

    def _stop_server(self):
        """
        Stops BT server process, releases network resources.
        """
        if self.server:

            if self._force_control_mode():
                # In case server is running and client side is ok:
                self.socket.send_pyobj({'ctrl': '_stop'})
                self.server_response = self.socket.recv_pyobj()

            else:
                self.server.terminate()
                self.server.join()
                self.server_response = 'Server process terminated.'

            self.log.info('{} Exit code: {}'.format(self.server_response,
                                                    self.server.exitcode))

        # Release client-side, if any:
        if self.context:
            self.context.destroy()

    def _force_control_mode(self):
        """
        Puts BT server to control mode.
        """
        # Check is there any faults with server process and connection?
        network_error = [
            (not self.server or not self.server.is_alive(), 'No running server found. Hint: forgot to call reset()?'),
            (not self.context or self.context.closed, 'No network connection found.'),
        ]
        for (err, msg) in network_error:
            if err:
                self.log.info(msg)
                self.server_response = msg
                return False

            # If everything works, insist to go 'control':
            self.server_response = {}
            attempt = 0

            while not 'ctrl' in self.server_response:
                self.socket.send_pyobj({'ctrl': '_done'})
                self.server_response = self.socket.recv_pyobj()
                attempt += 1
                self.log.debug('FORCE CONTROL MODE attempt: {}.\nResponse: {}'.format(attempt, self.server_response))

            return True

    def _assert_response(self, response):
        """
        Simple watcher:
        roughly checks if we really talking to environment (== episode is running).
        Rises exception if response given is not as expected.
        """
        if type(response) == tuple and len(response) == 4:
            pass

        else:
            msg = 'Unexpected environment response: {}\nHint: Forgot to call reset()?'.format(response)
            raise AssertionError(msg)

        self.log.debug('Env response checker received:\n{}\nas type: {}'.
                       format(response, type(response)))

    def _reset(self, state_only=True):  # By default, returns only initial state observation (Gym convention).
        """
        Implementation of OpenAI Gym env.reset method.
        'Rewinds' backtrader server and starts new episode
        within randomly selected time period.
        """

        # Server process check:
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()

        if self._force_control_mode():
            self.socket.send_pyobj({'ctrl': '_reset'})
            self.server_response = self.socket.recv_pyobj()

            # Get initial environment response:
            self.env_response = self._step(0)

            # Check (once) if state_space is as expected:
            if self.env_response[0].shape == self.observation_space.shape:
                pass

            else:
                msg = (
                    '\nState observation shape mismatch!\n' +
                    'Shape set by env: {},\n' +
                    'Shape returned by server: {}.\n' +
                    'Hint: Wrong Strategy.get_state() parameters?'
                ).format(self.observation_space.shape, self.env_response[0].shape)
                self.log.info(msg)
                self._stop_server()
                raise AssertionError(msg)

            if state_only:
                return self.env_response[0]
            else:
                return self.env_response

        else:
            msg = 'Something went wrong. env.reset() can not get response from server.'
            self.log.info(msg)
            raise ChildProcessError(msg)

    def _step(self, action):
        """
        Implementation of OpenAI Gym env.step method.
        Relies on remote backtrader server for actual environment dynamics computing.
        """
        # Are you in the list, ready to go and all that?
        if self.action_space.contains(action)\
            and not self._closed\
            and (self.socket is not None)\
            and not self.socket.closed:
            pass

        else:
            msg = (
                '\nAt least one of these is true:\n' +
                'Action error: (space is {}, action sent is {}): {}\n' +
                'Environment closed: {}\n' +
                'Network error [socket doesnt exists or closed]: {}\n' +
                'Hint: forgot to call reset()?'
            ).format(
                self.action_space, action, not self.action_space.contains(action),
                self._closed,
                not self.socket or self.socket.closed,
            )
            self.log.info(msg)
            raise AssertionError(msg)

        # Send action to backtrader engine, receive environment response
        self.socket.send_pyobj({'action': self.server_actions[action]})
        self.env_response = self.socket.recv_pyobj()

        # Is it?
        self._assert_response(self.env_response)

        return self.env_response

    def _close(self):
        """
        Implementation of OpenAI Gym env.close method.
        Puts BTgym server in Control Mode:
        """
        self._stop_server()
        self.log.debug('Environment closed.')

    def get_stat(self):
        """
        Returns last episode statistics.
        Note: when invoked, forces running episode to terminate.
        """
        if self._force_control_mode():
            self.socket.send_pyobj({'ctrl': '_getstat'})
            return self.socket.recv_pyobj()

        else:
            return self.server_response

    def _render(self, mode='other_mode', close=False):
        """
        Implementation of OpenAI Gym env.render method.
        Visualises current environment state.
        Takes `mode` key argument, returns image as rgb_array :
        `human` - current state observation as price lines;
        `agent` - current processed observation state as RL agent sees it;
        `episode` - plotted results of last completed episode.
        """
        if close:
            return None

        if not self._closed\
            and self.socket\
            and not self.socket.closed:
            pass

        else:
            msg = (
                '\nCan''t get renderings.'
                '\nAt least one of these is true:\n' +
                'Environment closed: {}\n' +
                'Network error [socket doesnt exists or closed]: {}\n' +
                'Hint: forgot to call reset()?'
            ).format(
                self._closed,
                not self.socket or self.socket.closed,
            )
            self.log.warning(msg)
            return None

        self.socket.send_pyobj({'ctrl': '_render', 'mode': mode})
        rgb_array = self.socket.recv_pyobj()

        self.rendered_rgb[mode] = rgb_array

        return rgb_array

    def stop(self):
        """
        Finishes current episode if any, does nothing otherwise.
        Leaves server running.
        """
        if self._force_control_mode():
            self.log.info('Episode stop forced.')

    def _restart_server(self):
        """
        Restarts server.
        """
        self._stop_server()
        self._start_server()