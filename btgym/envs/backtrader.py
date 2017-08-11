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
# logging.basicConfig(format='%(name)s: %(message)s')
import time
import zmq
import os
import itertools
# import psutil

import gym
from gym import error, spaces
#from gym import utils
#from gym.utils import seeding, closer

import backtrader as bt

from btgym import BTgymServer, BTgymStrategy, BTgymDataset, BTgymRendering, BTgymDataFeedServer

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

    # Datafeed Server management:
    data_master = True
    data_network_address = 'tcp://127.0.0.1:'  # using localhost.
    data_port = 4999
    data_server = None
    data_server_pid = None
    data_context = None
    data_socket = None
    data_server_response = None

    # Dataset:
    dataset = None  # BTgymDataset instance.
    dataset_stat = None

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
    server_response = None

    # Connection timeout:
    connect_timeout = 60  # server connection timeout in seconds.
    connect_timeout_step = 0.01  # time between retries in seconds.

    # Rendering:
    render_enabled = True
    renderer = None  # Rendering support.
    rendered_rgb = dict()  # Keep last rendered images for each mode.

    # Logging:
    log = None
    verbose = 0  # verbosity mode: 0 - silent, 1 - info, 2 - debugging level (lot of traffic!).

    closed = True

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
        self.data_network_address += str(self.data_port)

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

        if self.data_master:
            # DATASET preparation, only data_master executes this:
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

        # Connect/Start data server (and get dataset statistic):
        self.log.info('Connecting data_server...')
        self._start_data_server()
        self.log.info('...done.')
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
        # the only sensible way is to infer from raw Dataset price values (we already got those from data_server):
        if self.params['strategy']['state_low'] is None or self.params['strategy']['state_high'] is None:

            # Exclude 'volume' from columns we count:
            self.dataset_columns.remove('volume')

            # Override with absolute price min and max values:
            self.params['strategy']['state_low'] =\
            self.engine.strats[0][0][2]['state_low'] =\
                self.dataset_stat.loc['min', self.dataset_columns].min()

            self.params['strategy']['state_high'] =\
            self.engine.strats[0][0][2]['state_high'] =\
                self.dataset_stat.loc['max', self.dataset_columns].max()

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

        # If instance is datamaster - it may or may not want to launch self BTgymServer (can do it later via reset);
        # else it always need to launch it:
        #if not self.data_master:
        self._start_server()
        self.closed = False

        self.log.info('Environment is ready.')

    def _comm_with_timeout(self, socket, message, timeout, connect_timeout_step=0.01,):
        """
        Exchanges messages via socket, timeout sensitive.
        # Args:
            socket: zmq connected socket to communicate via;
            message: message to send;
            timeout: max time to wait for response;
            connect_timeout_step: time increments between retries.
        # Returns:
            dictionary:
                status: communication result;
                message: received message if status == `ok` or None;
                time: remote side response time.
        """
        response=dict(
            status='ok',
            message=None,
        )
        try:
            socket.send_pyobj(message)

        except:
            response['status'] = 'send_failed'
            return response

        for i in itertools.count():
            try:
                response['message'] = socket.recv_pyobj(flags=zmq.NOBLOCK)
                response['time'] = i * connect_timeout_step
                break

            except:
                time.sleep(connect_timeout_step)
                if i >= timeout / connect_timeout_step:
                    response['status'] = 'receive_failed'
                    return response

        return response

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
        self.server = BTgymServer(cerebro=self.engine,
                                  render=self.renderer,
                                  network_address=self.network_address,
                                  data_network_address=self.data_network_address,
                                  connect_timeout=self.connect_timeout,
                                  log=self.log)
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup:
        time.sleep(1)

        # Check connection:
        self.log.debug('Server started, pinging {} ...'.format(self.network_address))

        self.server_response = self._comm_with_timeout(
            socket=self.socket,
            message={'ctrl': 'ping!'},
            timeout=self.connect_timeout,
        )
        if self.server_response['status'] in 'ok':
            self.log.debug('Server seems ready with response: <{}>'.
                           format(self.server_response['message']))

        else:
            msg = 'Server unreachable with status: <{}>.'.format(self.server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

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

            while 'ctrl' not in self.server_response:
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
        # Data Server check:
        if self.data_master:
            if not self.data_server or not self.data_server.is_alive():
                self.log.info('No running data_server found, starting...')
                self._start_data_server()

        # Server process check:
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()

        if self._force_control_mode():
            self.socket.send_pyobj({'ctrl': '_reset'})
            self.server_response = self.socket.recv_pyobj()

            # Get initial environment response:
            self.env_response = self._step(0)

            # Check (once) if it is really (o,r,d,i) tuple:
            self._assert_response(self.env_response)

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
                self.log.error(msg)
                self._stop_server()
                raise AssertionError(msg)

            if state_only:
                return self.env_response[0]
            else:
                return self.env_response

        else:
            msg = 'Something went wrong. env.reset() can not get response from server.'
            self.log.error(msg)
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
        env_response = self._comm_with_timeout(
            socket=self.socket,
            message={'action': self.server_actions[action]},
            timeout=self.connect_timeout,
        )
        if not env_response['status'] in 'ok':
            msg = 'Env.step: server unreachable with status: <{}>.'.format(env_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        self.env_response = env_response ['message']

        return self.env_response

    def _close(self):
        """
        Implementation of OpenAI Gym env.close method.
        Puts BTgym server in Control Mode:
        """
        self._stop_server()
        self._stop_data_server()
        self.log.info('Environment closed.')

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
        self.log.info('Server restarted.')

    def _start_data_server(self):
        """
        For data_master environment:
            - configures backtrader REQ/REP server instance and starts server process.
        For others:
            - establishes network connection to existing data_server.
        """
        self.data_server = None

        # Ensure network resources:
        # 1. Release client-side, if any:
        if self.data_context:
            self.data_context.destroy()
            self.data_socket = None

        # Only data_master launches/stops data_server process:
        if self.data_master:
            # 2. Kill any process using server port:
            cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.data_port)
            os.system(cmd)

            # Configure and start server:
            self.data_server = BTgymDataFeedServer(
                dataset=self.dataset,
                network_address=self.data_network_address,
                log=self.log,
            )
            self.data_server.daemon = False
            self.data_server.start()
            # Wait for server to startup
            time.sleep(1)

        # Set up client channel:
        self.data_context = zmq.Context()
        self.data_socket = self.data_context.socket(zmq.REQ)
        self.data_socket.connect(self.data_network_address)

        # Check connection:
        self.log.debug('Pinging data_server at: {} ...'.format(self.data_network_address))

        self.data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': 'ping!'},
            timeout=self.connect_timeout,
        )
        if self.data_server_response['status'] in 'ok':
            self.log.debug('Data_server seems ready with response: <{}>'.
                          format(self.data_server_response['message']))

        else:
            msg = 'Data_server unreachable with status: <{}>.'.\
                format(self.data_server_response['status'])
            self.log.error(msg)
            raise SystemExit(msg)

        # Get info and statistic:
        self.dataset_stat, self.dataset_columns, self.data_server_pid = self._get_dataset_info()

    def _stop_data_server(self):
        """
        For data_master:
            - stops BT server process, releases network resources.
        """
        if self.data_master:
            if self.data_server is not None and self.data_server.is_alive():
                # In case server is running and is ok:
                self.data_socket.send_pyobj({'ctrl': '_stop'})
                self.data_server_response = self.data_socket.recv_pyobj()

            else:
                self.data_server.terminate()
                self.data_server.join()
                self.data_server_response = 'Data_server process terminated.'

            self.log.info('{} Exit code: {}'.format(self.data_server_response, self.data_server.exitcode))

        #if self.data_context:
        #    self.data_context.destroy()
        #    self.data_socket = None

    def _restart_data_server(self):
        """
        Restarts data_server.
        """
        self._stop_data_server()
        self._start_data_server()

    def _get_dataset_info(self):
        """
        Retrieves dataset descriptive statistic'.
        """
        self.data_socket.send_pyobj({'ctrl': '_get_info'})
        self.data_server_response = self.data_socket.recv_pyobj()

        return self.data_server_response['dataset_stat'],\
               self.data_server_response['dataset_columns'],\
               self.data_server_response['pid']
# asynchronous...