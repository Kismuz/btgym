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

from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys
import time
import zmq
import os
import copy
import numpy as np
import gym
from gym import spaces

from collections import OrderedDict

import backtrader as bt

from btgym import BTgymServer, BTgymBaseStrategy, BTgymDataset2, BTgymRendering, BTgymDataFeedServer
from btgym import DictSpace, ActionDictSpace
from btgym.datafeed.multi import BTgymMultiData

from btgym.rendering import BTgymNullRendering


############################## OpenAI Gym Environment  ##############################


class BTgymEnv(gym.Env):
    """
    Base OpenAI Gym API shell for Backtrader backtesting/trading library.
    """
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
    # connect_timeout_step = 0.01  # time between retries in seconds.

    # Rendering:
    render_enabled = True
    render_modes = ['human', 'episode', ]
    # `episode` - plotted episode results.
    # `human` - raw_state observation in conventional human-readable format.
    #  <obs_space_key> - rendering of arbitrary state presented in observation_space with same key.

    renderer = None  # Rendering support.
    rendered_rgb = dict()  # Keep last rendered images for each mode.

    # Logging and id:
    log = None
    log_level = None  # logbook level: NOTICE, WARNING, INFO, DEBUG etc. or its integer equivalent;
    verbose = 0  # verbosity mode, valid only if no `log_level` arg has been provided:
    # 0 - WARNING, 1 - INFO, 2 - DEBUG.
    task = 0
    asset_names = ('default_asset',)
    data_lines_names = ('default_asset',)
    cash_name = 'default_cash'

    random_seed = None

    closed = True

    def __init__(self, **kwargs):
        """

        Keyword Args:
            dataset (btgym.datafeed):                       BTgymDataDomain instance
            strategy=None (btgym.startegy):                 strategy to be used by `engine`, any subclass of
                                                            btgym.strategy.base.BTgymBaseStrateg
            engine=None (bt.Cerebro):                       environment simulation engine, any bt.Cerebro subclass,
                                                            overrides `strategy` arg.
            network_address=`tcp://127.0.0.1:` (str):       BTGym_server address.
            port=5500 (int):                                network port to use for server - API_shell communication.
            data_master=True (bool):                        let this environment control over data_server;
            data_network_address=`tcp://127.0.0.1:` (str):  data_server address.
            data_port=4999 (int):                           network port to use for server -- data_server communication.
            connect_timeout=60 (int):                       server connection timeout in seconds.
            render_enabled=True (bool):                     enable rendering for this environment;
            render_modes=['human', 'episode'] (list):       `episode` - plotted episode results;
                                                            `human` - raw_state observation.
            **render_args (any):                            any render-related args, passed through to renderer class.
            verbose=0 (int):                                verbosity mode, {0 - WARNING, 1 - INFO, 2 - DEBUG}
            log_level=None (int):                           logbook level {DEBUG=10, INFO=11, NOTICE=12, WARNING=13},
                                                            overrides `verbose` arg;
            log=None (logbook.Logger):                      external logbook logger,
                                                            overrides `log_level` and `verbose` args.
            task=0 (int):                                   environment id
            random_seed(int):                               numpy random seed, def: None

        Environment kwargs applying logic::

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
                override corresponding default parameter.
        """
        # Parameters and default values:
        self.params = dict(

            # Backtrader engine mandatory parameters:
            engine=dict(
                start_cash=100.0,  # initial trading capital.
                broker_commission=0.001,  # trade execution commission, default is 0.1% of operation value.
                fixed_stake=10,  # single trade stake is fixed type by def.
            ),
            # Dataset mandatory parameters:
            dataset=dict(
                filename=None,
            ),
            strategy=dict(
                state_shape=dict(),
            ),
            render=dict(),
        )

        # Update self attributes, remove used kwargs:
        for key in dir(self):
            if key in kwargs.keys():
                setattr(self, key, kwargs.pop(key))

        self.metadata = {'render.modes': self.render_modes}

        # Logging and verbosity control:
        if self.log is None:
            StreamHandler(sys.stdout).push_application()
            if self.log_level is None:
                log_levels = [(0, NOTICE), (1, INFO), (2, DEBUG)]
                self.log_level = WARNING
                for key, value in log_levels:
                    if key == self.verbose:
                        self.log_level = value
            self.log = Logger('BTgymAPIshell_{}'.format(self.task), level=self.log_level)

        # Random seeding:
        np.random.seed(self.random_seed)

        # Network parameters:
        self.network_address += str(self.port)
        self.data_network_address += str(self.data_port)

        # Set server rendering:
        if self.render_enabled:
            self.renderer = BTgymRendering(self.metadata['render.modes'], log_level=self.log_level, **kwargs)

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

        # Disable multiply data streams (multi-assets) [for data-master]:
        try:
            assert not isinstance(self.dataset, BTgymMultiData)

        except AssertionError:
            self.log.error(
                'Using multiply data streams with base BTgymEnv class not supported. Use designated class.'
            )
            raise ValueError

        if self.data_master:  # TODO: remove as data preparation is a responsibility of another class.
            # DATASET preparation, only data_master executes this:
            if self.dataset is None:
                raise AttributeError()

        # Connect/Start data server (and get dataset configuration and statistic):
        self.log.info('Connecting data_server...')
        self._start_data_server()
        self.log.info('...done.')

        # After starting data-server we have self.data_names attribute filled.

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
                self.strategy = BTgymBaseStrategy
                msg2 = 'Base Strategy class used.'

            # Add, using kwargs on top of defaults:
            # self.log.debug('kwargs for strategy: {}'.format(kwargs))
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
            self.params['strategy'][t_key] = t_value

        # Update it with values from strategy 'passed-to params':
        for key, value in self.engine.strats[0][0][2].items():
            self.params['strategy'][key] = value

        self.asset_names = self.params['strategy']['asset_names']
        self.server_actions = {name: self.params['strategy']['portfolio_actions'] for name in self.asset_names}
        self.cash_name = self.params['strategy']['cash_name']

        self.params['strategy']['initial_action'] = self.get_initial_action()
        self.params['strategy']['initial_portfolio_action'] = self.get_initial_portfolio_action()

        # Only single  asset is supported by base class:
        try:
            assert len(list(self.asset_names)) == 1

        except AssertionError:
            self.log.error(
                'Using multiply assets with base BTgymEnv class not supported. Use designated class.'
            )
            raise ValueError

        try:
            assert set(self.asset_names).issubset(set(self.data_lines_names))

        except AssertionError:
            msg = 'Assets names should be subset of data_lines names, but got: assets: {}, data_lines: {}'.format(
                set(self.asset_names), set(self.data_lines_names)
            )
            self.log.error(msg)
            raise ValueError(msg)

        # ... Push it all back (don't ask):
        for key, value in self.params['strategy'].items():
            self.engine.strats[0][0][2][key] = value

        # For 'raw_state' min/max values,
        # the only way is to infer from raw Dataset price values (we already got those from data_server):
        if 'raw' in self.params['strategy']['state_shape'].keys():
            # Exclude 'volume' from columns we count:
            self.dataset_columns.remove('volume')

            # print(self.params['strategy'])
            # print('self.engine.strats[0][0][2]:', self.engine.strats[0][0][2])
            # print('self.engine.strats[0][0][0].params:', self.engine.strats[0][0][0].params._gettuple())

            # Override with absolute price min and max values:
            self.params['strategy']['state_shape']['raw'].low = \
                self.engine.strats[0][0][2]['state_shape']['raw'].low = \
                np.zeros(self.params['strategy']['state_shape']['raw'].shape) + \
                self.dataset_stat.loc['min', self.dataset_columns].min()

            self.params['strategy']['state_shape']['raw'].high = \
                self.engine.strats[0][0][2]['state_shape']['raw'].high = \
                np.zeros(self.params['strategy']['state_shape']['raw'].shape) + \
                self.dataset_stat.loc['max', self.dataset_columns].max()

            self.log.info('Inferring `state[raw]` high/low values form dataset: {:.6f} / {:.6f}.'.
                          format(self.dataset_stat.loc['min', self.dataset_columns].min(),
                                 self.dataset_stat.loc['max', self.dataset_columns].max()))

        # Set observation space shape from engine/strategy parameters:
        self.observation_space = DictSpace(self.params['strategy']['state_shape'])

        self.log.debug('Obs. shape: {}'.format(self.observation_space.spaces))
        # self.log.debug('Obs. min:\n{}\nmax:\n{}'.format(self.observation_space.low, self.observation_space.high))

        # Set action space (one-key dict for this class) and corresponding server messages:
        self.action_space = ActionDictSpace(
            base_actions=self.params['strategy']['portfolio_actions'],
            assets=self.asset_names
        )

        # Finally:
        self.server_response = None
        self.env_response = None

        self._start_server()
        self.closed = False

        self.log.info('Environment is ready.')

    def _seed(self, seed=None):
        """
        Sets env. random seed.

        Args:
            seed:   int or None
        """
        self.random_seed = seed
        np.random.seed(self.random_seed)

    @staticmethod
    def _comm_with_timeout(socket, message, ):
        """
        Exchanges messages via socket, timeout sensitive.

        Args:
            socket: zmq connected socket to communicate via;
            message: message to send;

        Note:
            socket zmq.RCVTIMEO and zmq.SNDTIMEO should be set to some finite number of milliseconds.

        Returns:
            dictionary:
                `status`: communication result;
                `message`: received message if status == `ok` or None;
                `time`: remote side response time.
        """
        response = dict(
            status='ok',
            message=None,
        )
        try:
            socket.send_pyobj(message)

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                response['status'] = 'send_failed_due_to_connect_timeout'

            else:
                response['status'] = 'send_failed_for_unknown_reason'
            return response

        start = time.time()
        try:
            response['message'] = socket.recv_pyobj()
            response['time'] = time.time() - start

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                response['status'] = 'receive_failed_due_to_connect_timeout'

            else:
                response['status'] = 'receive_failed_for_unknown_reason'
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
        self.socket.setsockopt(zmq.RCVTIMEO, self.connect_timeout * 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, self.connect_timeout * 1000)
        self.socket.connect(self.network_address)

        # Configure and start server:
        self.server = BTgymServer(
            cerebro=self.engine,
            render=self.renderer,
            network_address=self.network_address,
            data_network_address=self.data_network_address,
            connect_timeout=self.connect_timeout,
            log_level=self.log_level,
            task=self.task,
        )
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup:
        time.sleep(1)

        # Check connection:
        self.log.info('Server started, pinging {} ...'.format(self.network_address))

        self.server_response = self._comm_with_timeout(
            socket=self.socket,
            message={'ctrl': 'ping!'}
        )
        if self.server_response['status'] in 'ok':
            self.log.info('Server seems ready with response: <{}>'.
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
            self.socket = None

    def _force_control_mode(self):
        """Puts BT server to control mode.
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
        try:
            assert type(response) == tuple and len(response) == 4

        except AssertionError:
            msg = 'Unexpected environment response: {}\nHint: Forgot to call reset() or reset_data()?'.format(response)
            self.log.exception(msg)
            raise AssertionError(msg)

        self.log.debug('Response checker received:\n{}\nas type: {}'.
                       format(response, type(response)))

    def _print_space(self, space, _tab=''):
        """
        Parses observation space shape or response.

        Args:
            space: gym observation space or state.

        Returns:
            description as string.
        """
        response = ''
        if type(space) in [dict, OrderedDict]:
            for key, value in space.items():
                response += '\n{}{}:{}\n'.format(_tab, key, self._print_space(value, '   '))

        elif type(space) in [spaces.Dict, DictSpace]:
            for s in space.spaces:
                response += self._print_space(s, '   ')

        elif type(space) in [tuple, list]:
            for i in space:
                response += self._print_space(i, '   ')

        elif type(space) == np.ndarray:
            response += '\n{}array of shape: {}, low: {}, high: {}'.format(_tab, space.shape, space.min(), space.max())

        else:
            response += '\n{}{}, '.format(_tab, space)
            try:
                response += 'low: {}, high: {}'.format(space.low.min(), space.high.max())

            except (KeyError, AttributeError, ArithmeticError, ValueError) as e:
                pass
                # response += '\n{}'.format(e)

        return response

    def get_initial_action(self):
        return {asset: 0 for asset in self.asset_names}

    def get_initial_portfolio_action(self):
        return {asset: actions[0] for asset, actions in self.server_actions.items()}

    def reset(self, **kwargs):
        """
        Implementation of OpenAI Gym env.reset method. Starts new episode. Episode data are sampled
        according to data provider class logic, controlled via kwargs. Refer `BTgym_Server` and data provider
        classes for details.

        Args:
            kwargs:         any kwargs; this dictionary is passed through to BTgym_server side without any checks and
                            modifications; currently used for data sampling control;

        Returns:
            observation space state

        Notes:
            Current kwargs accepted is::


                episode_config=dict(
                    get_new=True,
                    sample_type=0,
                    b_alpha=1,
                    b_beta=1
                ),
                trial_config=dict(
                    get_new=True,
                    sample_type=0,
                    b_alpha=1,
                    b_beta=1
                )

        """
        # Data Server check:
        if self.data_master:
            if not self.data_server or not self.data_server.is_alive():
                self.log.info('No running data_server found, starting...')
                self._start_data_server()

            # Domain dataset status check:
            self.data_server_response = self._comm_with_timeout(
                socket=self.data_socket,
                message={'ctrl': '_get_info'}
            )
            if not self.data_server_response['message']['dataset_is_ready']:
                self.log.info(
                    'Data domain `reset()` called prior to `reset_data()` with [possibly inconsistent] defaults.'
                )
                self.reset_data()

        # Server process check:
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()

        if self._force_control_mode():
            self.server_response = self._comm_with_timeout(
                socket=self.socket,
                message={'ctrl': '_reset', 'kwargs': kwargs}
            )
            # Get initial environment response:
            self.env_response = self.step(self.get_initial_action())

            # Check (once) if it is really (o,r,d,i) tuple:
            self._assert_response(self.env_response)

            # Check (once) if state_space is as expected:
            try:
                assert self.observation_space.contains(self.env_response[0])

            except (AssertionError, AttributeError) as e:
                msg1 = self._print_space(self.observation_space.spaces)
                msg2 = self._print_space(self.env_response[0])
                msg3 = ''
                for step_info in self.env_response[-1]:
                    msg3 += '{}\n'.format(step_info)
                msg = (
                        '\nState observation shape/range mismatch!\n' +
                        'Space set by env: \n{}\n' +
                        'Space returned by server: \n{}\n' +
                        'Full response:\n{}\n' +
                        'Reward: {}\n' +
                        'Done: {}\n' +
                        'Info:\n{}\n' +
                        'Hint: Wrong Strategy.get_state() parameters?'
                ).format(
                    msg1,
                    msg2,
                    self.env_response[0],
                    self.env_response[1],
                    self.env_response[2],
                    msg3,
                )
                self.log.exception(msg)
                self._stop_server()
                raise AssertionError(msg)

            return self.env_response[0]


        else:
            msg = 'Something went wrong. env.reset() can not get response from server.'
            self.log.exception(msg)
            raise ChildProcessError(msg)

    def step(self, action):
        """
        Implementation of OpenAI Gym env.step() method.
        Makes a step in the environment.

        Args:
            action:     int or dict, action compatible to env.action_space

        Returns:
            tuple (Observation, Reward, Info, Done)

        """
        # If we got int as action - try to treat it as an action for single-valued action space dict:
        self.log.debug('got action: {} as {}'.format(action, type(action)))

        # Are you in the list, ready to go and all that?
        if not self.action_space.contains(action):
            # If action received as scalar - try to convert it to action space:
            try:
                a = copy.deepcopy(int(action))

            except Exception as e:
                self.log.error('Received value {} can not be converted to member of action space.'.format(action))
                raise e

            action = {key: a for key in self.action_space.spaces.keys()}

            try:
                # Check again:
                assert self.action_space.contains(action)

            except AssertionError as e:
                self.log.error('Action from scalar {} --> {} is out of action space.'.format(a, action))
                raise e

            self.log.debug('got action as scalar: {}, converted to: {}'.format(a, action))

        if not self._closed \
                and (self.socket is not None) \
                and not self.socket.closed:
            pass

        else:
            msg = (
                    '\nAt least one of these is true:\n' +
                    'Environment closed: {}\n' +
                    'Network error [socket doesnt exists or closed]: {}\n' +
                    'Hint: forgot to call reset() or out of action space?'
            ).format(
                self._closed,
                not self.socket or self.socket.closed,
            )
            self.log.error(msg)
            raise ConnectionError(msg)

        # Send action (as dict of strings) to backtrader engine, receive environment response:
        action_as_dict = {key: self.server_actions[key][value] for key, value in action.items()}
        # print('step: ', action, action_as_dict)
        env_response = self._comm_with_timeout(
            socket=self.socket,
            message={'action': action_as_dict}
        )
        if not env_response['status'] in 'ok':
            msg = '.step(): server unreachable with status: <{}>.'.format(env_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        self.env_response = env_response['message']

        return self.env_response

    def close(self):
        """
        Implementation of OpenAI Gym env.close method.
        Puts BTgym server in Control Mode.
        """
        self.log.debug('close.call()')
        self._stop_server()
        self._stop_data_server()
        self.log.info('Environment closed.')

    def get_stat(self):
        """
        Returns last run episode statistics.

        Note:
            when invoked, forces running episode to terminate.
        """
        if self._force_control_mode():
            self.socket.send_pyobj({'ctrl': '_getstat'})
            return self.socket.recv_pyobj()

        else:
            return self.server_response

    def render(self, mode='other_mode', close=False):
        """
        Implementation of OpenAI Gym env.render method.
        Visualises current environment state.

        Args:
            `mode`:     str, any of these::

                            `human` - current state observation as price lines;
                            `episode` - plotted results of last completed episode.
                            [other_key] - corresponding to any custom observation space key
        """
        if close:
            return None

        if not self._closed \
                and self.socket \
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
        if mode not in self.render_modes:
            raise ValueError('Unexpected render mode {}'.format(mode))
        self.socket.send_pyobj({'ctrl': '_render', 'mode': mode})

        rgb_array_dict = self.socket.recv_pyobj()

        self.rendered_rgb.update(rgb_array_dict)

        return self.rendered_rgb[mode]

    def _stop(self):
        """
        Finishes current episode if any, does nothing otherwise. Leaves server running.
        """
        if self._force_control_mode():
            self.log.info('Episode stop forced.')

    def _restart_server(self):
        """Restarts server.
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
                log_level=self.log_level,
                task=self.task
            )
            self.data_server.daemon = False
            self.data_server.start()
            # Wait for server to startup
            time.sleep(1)

        # Set up client channel:
        self.data_context = zmq.Context()
        self.data_socket = self.data_context.socket(zmq.REQ)
        self.data_socket.setsockopt(zmq.RCVTIMEO, self.connect_timeout * 1000)
        self.data_socket.setsockopt(zmq.SNDTIMEO, self.connect_timeout * 1000)
        self.data_socket.connect(self.data_network_address)

        # Check connection:
        self.log.debug('Pinging data_server at: {} ...'.format(self.data_network_address))

        self.data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': 'ping!'}
        )
        if self.data_server_response['status'] in 'ok':
            self.log.debug('Data_server seems ready with response: <{}>'.
                           format(self.data_server_response['message']))

        else:
            msg = 'Data_server unreachable with status: <{}>.'. \
                format(self.data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        # Get info and statistic:
        self.dataset_stat, self.dataset_columns, self.data_server_pid, self.data_lines_names = self._get_dataset_info()

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

        if self.data_context:
            self.data_context.destroy()
            self.data_socket = None

    def _restart_data_server(self):
        """
        Restarts data_server.
        """
        if self.data_master:
            self._stop_data_server()
            self._start_data_server()

    def _get_dataset_info(self):
        """
        Retrieves dataset configuration and descriptive statistic.
        """
        self.data_socket.send_pyobj({'ctrl': '_get_info'})
        self.data_server_response = self.data_socket.recv_pyobj()

        return self.data_server_response['dataset_stat'], \
               self.data_server_response['dataset_columns'], \
               self.data_server_response['pid'], \
               self.data_server_response['data_names']

    def reset_data(self, **kwargs):
        """
        Resets data provider class used, whatever it means for that class. Gets data_server ready to provide data.
        Supposed to be called before first env.reset().

        Note:
            when invoked, forces running episode to terminate.

        Args:
            **kwargs:   data provider class .reset() method specific.
        """
        if self.closed:
            self._start_server()
            if self.data_master:
                self._start_data_server()
            self.closed = False

        else:
            _ = self._force_control_mode()

        if self.data_master:
            if self.data_server is None or not self.data_server.is_alive():
                self._restart_data_server()

            self.data_server_response = self._comm_with_timeout(
                socket=self.data_socket,
                message={'ctrl': '_reset_data', 'kwargs': kwargs}
            )
            if self.data_server_response['status'] in 'ok':
                self.log.debug('Dataset seems ready with response: <{}>'.
                               format(self.data_server_response['message']))

            else:
                msg = 'Data_server unreachable with status: <{}>.'. \
                    format(self.data_server_response['status'])
                self.log.error(msg)
                raise SystemExit(msg)

        else:
            pass
