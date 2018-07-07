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
import itertools
import copy
import numpy as np

from gym import spaces
import backtrader as bt

from btgym import BTgymRendering #, DictSpace
from btgym.datafeed.multi import BTgymMultiData

from btgym.rendering import BTgymNullRendering

from btgym.envs.base import BTgymEnv


class MultiDiscreteEnv(BTgymEnv):
    """
    OpenAI Gym API shell for Backtrader backtesting/trading library with multiply data streams (assets) support.
    Action space is dictionary of discrete actions for every asset.
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

    closed = True

    def __init__(self, dataset, engine,  **kwargs):
        """
        This class requires dataset, strategy, engine instances to be passed explicitly.

        Args:

            dataset(btgym.datafeed):                        BTgymDataDomain instance;
            engine(bt.Cerebro):                             environment simulation engine, any bt.Cerebro subclass,
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


        """
        self.dataset = dataset
        self.engine = engine
        # Parameters and default values:
        self.params = dict(
            engine={},
            dataset={},
            strategy={},
            render={},
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
            self.log = Logger('BTgymMultiAssetShell_{}'.format(self.task), level=self.log_level)

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

        # We require dataset instance to have `data_config` attribute to infer assets names:
        try:
            self.assets = [key for key in self.dataset.data_config.keys()]

        except AttributeError:
            self.log.error('`data_config` attribute for dataset instance passed not found, cannot infer assets config.')
            raise AttributeError()

        if self.data_master:
            # Append logging:
            self.dataset.set_logger(self.log_level, self.task)

            # Update params -2: pull from dataset, remove used kwargs:
            self.params['dataset'].update(self.dataset.params)
            for key in self.params['dataset'].keys():
                if key in kwargs.keys():
                    _ = kwargs.pop(key)

        # Connect/Start data server (and get dataset statistic):
        self.log.info('Connecting data_server...')
        self._start_data_server()
        self.log.info('...done.')

        # Define observation space shape, minimum / maximum values and agent action space.
        # Retrieve values from configured engine or...

        # ...Update params -4:
        # Pull strategy defaults to environment params dict :
        for t_key, t_value in self.engine.strats[0][0][0].params._gettuple():
            self.params['strategy'][t_key] = t_value

        # Update it with values from strategy 'passed-to params':
        for key, value in self.engine.strats[0][0][2].items():
            self.params['strategy'][key] = value

        self.server_actions = {name: self.params['strategy']['portfolio_actions'] for name in self.assets}

        self.params['strategy']['initial_action'] = self.get_initial_action()
        self.params['strategy']['initial_portfolio_action'] = self.get_initial_portfolio_action()

        # ... Push it all back (don't ask):
        for key, value in self.params['strategy'].items():
            self.engine.strats[0][0][2][key] = value

        # For 'raw_state' min/max values,
        # the only way is to infer from raw Dataset price values (we already got those from data_server):
        if 'raw_state' in self.params['strategy']['state_shape'].keys():
            # Exclude 'volume' from columns we count:
            self.dataset_columns.remove('volume')

            # print(self.params['strategy'])
            # print('self.engine.strats[0][0][2]:', self.engine.strats[0][0][2])
            # print('self.engine.strats[0][0][0].params:', self.engine.strats[0][0][0].params._gettuple())

            # Override with absolute price min and max values:
            self.params['strategy']['state_shape']['raw_state'].low = \
                self.engine.strats[0][0][2]['state_shape']['raw_state'].low = \
                np.zeros(self.params['strategy']['state_shape']['raw_state'].shape) + \
                self.dataset_stat.loc['min', self.dataset_columns].min()

            self.params['strategy']['state_shape']['raw_state'].high = \
                self.engine.strats[0][0][2]['state_shape']['raw_state'].high = \
                np.zeros(self.params['strategy']['state_shape']['raw_state'].shape) + \
                self.dataset_stat.loc['max', self.dataset_columns].max()

            self.log.info('Inferring `state_raw` high/low values form dataset: {:.6f} / {:.6f}.'.
                          format(self.dataset_stat.loc['min', self.dataset_columns].min(),
                                 self.dataset_stat.loc['max', self.dataset_columns].max()))

        # Set observation space shape from engine/strategy parameters:
        self.observation_space = spaces.Dict(self.params['strategy']['state_shape'])

        self.log.debug('Obs. shape: {}'.format(self.observation_space.spaces))

        # Set action space and corresponding server messages:
        self.action_space = spaces.Dict(
            {
                name: spaces.Discrete(len(self.params['strategy']['portfolio_actions']))
                for name in self.assets
            }
        )
        self.log.debug('Act. space shape: {}'.format(self.action_space.spaces))

        # Finally:
        self.server_response = None
        self.env_response = None

        # if not self.data_master:
        self._start_server()
        self.closed = False

        self.log.info('Environment is ready.')

    def get_initial_action(self):
        return {key: 0 for key in self.assets}

    def get_initial_portfolio_action(self):
        return {key: self.server_actions[key][0] for key in self.assets}

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
        if isinstance(action, int) and len(list(self.action_space.spaces.keys())) == 1:
            a = copy.deepcopy(action)
            action = {key: a for key in self.action_space.spaces.keys()}

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
            self.log.exception(msg)
            raise AssertionError(msg)

        # Send action (as dict of strings) to backtrader engine, receive environment response:
        env_response = self._comm_with_timeout(
            socket=self.socket,
            message={'action': {key: self.server_actions[key][value] for key, value in action.items()}}
        )
        if not env_response['status'] in 'ok':
            msg = '.step(): server unreachable with status: <{}>.'.format(env_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        self.env_response = env_response ['message']

        return self.env_response