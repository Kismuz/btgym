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
import numpy as np

import backtrader as bt

from btgym import BTgymRendering, DictSpace, ActionDictSpace

from btgym.rendering import BTgymNullRendering

from btgym.envs.base import BTgymEnv


class MultiDiscreteEnv(BTgymEnv):
    """
    OpenAI Gym API shell for Backtrader backtesting/trading library with multiply data streams (assets) support.
    Action space is dictionary of discrete actions for every asset.

    Multi-asset setup explanation:

        1. This environment expects Dataset to be instance of `btgym.datafeed.multi.BTgymMultiData`, which sets
        number,  specifications and sampling synchronisation for historic data for all assets
        one want to trade jointly.

        2. Internally every episodic asset data is converted to single bt.feed and added to environment strategy
        as separate named data-line (see backtrader docs for extensive explanation of data-lines concept). Strategy is
        expected to properly handle all received data-lines.

        3. btgym.spaces.ActionDictSpace and order execution. Strategy expects to receive separate action
        for every asset in form of dictionary: `{asset_name_1: action, ..., asset_name_K: action}`
        for K assets added, and issues orders for all assets within a single strategy step.
        It is supposed that actions are discrete [for this environment] and same for every asset.
        Base actions are set by strategy.params.portfolio_actions, defaults are: ('hold', 'buy', 'sell', 'close') which
        equals to `gym.spaces.Discrete` with depth `N=4 (~number of actions: 0, 1, 2, 3)`.
        That is, for `K` assets environment action space will be a shallow dictionary `(DictSpace)` of discrete spaces:
        `{asset_name_1: gym.spaces.Discrete(N), ..., asset_name_K: gym.spaces.Discrete(N)}`

            Example::

                if datalines added via BTgymMultiData are: ['eurchf', 'eurgbp', 'eurgpy', 'eurusd'],
                and base asset actions are ['hold', 'buy', 'sell', 'close'], than:

                env.action.space will be:
                    DictSpace(
                        {
                            'eurchf': gym.spaces.Discrete(4),
                            'eurgbp': gym.spaces.Discrete(4),
                            'eurgpy': gym.spaces.Discrete(4),
                            'eurusd': gym.spaces.Discrete(4),
                        }
                    )
                single environment action instance (as seen inside strategy):
                    {
                        'eurchf': 'hold',
                        'eurgbp': 'buy',
                        'eurgpy': 'hold',
                        'eurusd': 'close',
                    }
                corresponding action integer encoding as passed to environment via .step():
                    {
                        'eurchf': 0,
                        'eurgbp': 1,
                        'eurgpy': 0,
                        'eurusd': 3,
                    }
                vector of integers (categorical):
                    (0, 1, 0, 3)

        4. Environment actions cardinality and encoding. Note that total set of environment actions for `K` assets
        and `N` base actions is a `cartesian product of K sets of N elements each`. It can be encoded as `vector of integers,
        single scalar, binary or one_hot`. As cardinality skyrockets with `K`, `multi-discrete` action setup is only suited
        for small number of assets.

            Example::

                Setup with 4 assets and 4 base actions [hold, buy, sell, close] spawns total of 256 possible
                environment actions expressed by single integer in [0, 255] or binary encoding:
                    vector str :                            vector:         int:   binary:
                    ('hold', 'hold', 'hold', 'hold')     -> (0, 0, 0, 0) -> 0   -> 00000000
                    ('hold', 'hold', 'hold', 'buy')      -> (0, 0, 0, 1) -> 1   -> 00000001
                    ...         ...         ...
                    ('close', 'close', 'close', 'sell')  -> (3, 3, 3, 2) -> 254 -> 11111110
                    ('close', 'close', 'close', 'close') -> (3, 3, 3, 3) -> 255 -> 11111111

        Internally there is some weirdness with encodings as we jump forth and back between
        dictionary of names or categorical encodings and binary encoding or one-hot encoding.
        As a rule: strategy operates with dictionary of string names of actions, environment sees action as dictionary
        of integer numbers while policy estimator operates with either binary or one-hot encoding.

        5. Observation space: is nested DictSpace, where 'external' part part of space should hold specifications for
        every asset added.

            Example::

                if datalines added via BTgymMultiData are:
                    'eurchf', 'eurgbp', 'eurgpy', 'eurusd';

                environment observation space should be DictSpace:
                 {
                    'raw': spaces.Box(low=-1000, high=1000, shape=(128, 4), dtype=np.float32),
                    'external': DictSpace(
                        {
                            'eurusd': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                            'eurgbp': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                            'eurchf': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                            'eurgpy': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                        }
                    ),
                    'internal': spaces.Box(...),
                    'datetime': spaces.Box(...),
                    'metadata': DictSpace(...)
                }

                refer to strategies declarations for full code.
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

    def __init__(self, engine, dataset=None, **kwargs):
        """
        This class requires dataset, strategy, engine instances to be passed explicitly.

        Args:
            dataset(btgym.datafeed):                        BTgymDataDomain instance;
            engine(bt.Cerebro):                             environment simulation engine, any bt.Cerebro subclass,

        Keyword Args:
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
            self.log = Logger('BTgymMultiDataShell_{}'.format(self.task), level=self.log_level)

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

        if self.data_master:
            try:
                assert self.dataset is not None

            except AssertionError:
                msg = 'Dataset instance shoud be provided for data_master environment.'
                self.log.error(msg)
                raise ValueError(msg)

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
        # After starting data-server we have self.assets attribute, dataset statisitc etc. filled.

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

        # Disabling this check allows derivative assets:

        # try:
        #     assert set(self.asset_names).issubset(set(self.data_lines_names))
        #
        # except AssertionError:
        #     msg = 'Assets names should be subset of data_lines names, but got: assets: {}, data_lines: {}'.format(
        #         set(self.asset_names), set(self.data_lines_names)
        #     )
        #     self.log.error(msg)
        #     raise ValueError(msg)

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
        self.observation_space = DictSpace(self.params['strategy']['state_shape'])

        self.log.debug('Obs. shape: {}'.format(self.observation_space.spaces))

        # Set action space and corresponding server messages:
        self.action_space = ActionDictSpace(
            base_actions=self.params['strategy']['portfolio_actions'],
            assets=self.asset_names
        )

        self.log.debug('Act. space shape: {}'.format(self.action_space.spaces))

        # Finally:
        self.server_response = None
        self.env_response = None

        # if not self.data_master:
        self._start_server()
        self.closed = False

        self.log.info('Environment is ready.')

