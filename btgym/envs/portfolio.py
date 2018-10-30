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
import copy

import backtrader as bt

from btgym import BTgymRendering, DictSpace, ActionDictSpace

from btgym.rendering import BTgymNullRendering

from btgym.envs.base import BTgymEnv


class PortfolioEnv(BTgymEnv):
    """
        OpenAI Gym API shell for Backtrader backtesting/trading library with multiply assets support.
        Action space is dictionary of contionious  actions for every asset.
        This setup closely relates to continuous portfolio optimisation problem definition.

        Setup explanation:

            0. Problem definition.
            Consider setup with one riskless asset acting as broker account cash and K (by default - one) risky assets.
            For every risky asset there exists track of historic price records referred as `data-line`.
            Apart from assets data lines there possibly exists number of exogenous data lines holding some
            information and statistics, e.g. economic indexes, encoded news, macroeconomic indicators, weather forecasts
            etc. which are considered relevant and valuable for decision-making.
            It is supposed for this setup that:
            i. there is no interest rate for base (riskless) asset;
            ii. short selling is not permitted;
            iii. transaction costs are modelled via broker commission;
            iv. 'market liquidity' and 'capital impact' assumptions are met;
            v. time indexes match for all data lines provided;

            1. Assets and datalines.
            This environment expects Dataset to be instance of `btgym.datafeed.multi.BTgymMultiData`, which sets
            number,  specifications and sampling synchronisation for historic data for all assets and data lines.

            Namely, one should define data_config dictionary of `data lines` and list of `assets`.
            `data_config` specifies all data sources used by strategy, while `assets` defines subset of `data lines`
            which is supposed to hold historic data for risky portfolio assets.

            Internally every episodic asset data is converted to single bt.feed and added to environment strategy
            as separate named data_line (see backtrader docs for extensive explanation of data_lines concept).
            Every non-asset data line as also added as bt.feed with difference that it is not 'tradable' i.e. it is
            impossible to issue trade orders on such line.
            Strategy is expected to properly handle all received data-lines.

                Example::

                    1. Four data streams added via Dataset.data_config,
                       portfolio consists of four assets, added via strategy_params, cash is EUR:

                        data_config = {
                            'usd': {'filename': '.../DAT_ASCII_EURUSD_M1_2017.csv'},
                            'gbp': {'filename': '.../DAT_ASCII_EURGBP_M1_2017.csv'},
                            'jpy': {'filename': '.../DAT_ASCII_EURJPY_M1_2017.csv'},
                            'chf': {'filename': '.../DAT_ASCII_EURCHF_M1_2017.csv'},
                        }
                        cash_name = 'eur'
                        assets_names = ['usd', 'gbp', 'jpy', 'chf']

                    2. Three streams added, only two of them form portfolio; DXY stream is `decision-making` only:
                        data_config = {
                            'usd': {'filename': '.../DAT_ASCII_EURUSD_M1_2017.csv'},
                            'gbp': {'filename': '.../DAT_ASCII_EURGBP_M1_2017.csv'},
                            '​DXY': {'filename': '.../DAT_ASCII_DXY_M1_2017.csv'},
                        }
                        cash_name = 'eur'
                        assets_names = ['usd', 'gbp']


            2. btgym.spaces.ActionDictSpace and order execution.
            ActionDictSpace is an extension of OpenAI Gym DictSpace providing domain-specific functionality.
            Strategy expects to receive separate action for every K+1 asset in form of dictionary:
            `{cash_name: a[0], asset_name_1: a[1], ..., asset_name_K: a[K]}` for K risky assets added,
            where base actions are real numbers: `a[i] in [0,1], 0<=i<=K, SUM{a[i]} = 1`. Whole action should be
            interpreted as order to adjust portfolio to have share `a[i] * 100% for i-th  asset`.

            Therefore, base actions are gym.spaces.Box and for K assets environment action space will be a shallow
            DictSpace of K+1 continuous spaces: `{cash_name: gym.spaces.Box(low=0, high=1),
            asset_name_1: gym.spaces.Box(low=0, high=1), ..., asset_name_K: gym.spaces.Box(low=0, high=1)}`

            3. TODO: refine order execution control, see: https://community.backtrader.com/topic/152/multi-asset-ranking-and-rebalancing/2?page=1

                Example::

                    if cash asset is 'eur',
                    risky assets added are: ['chf', 'gbp', 'gpy', 'usd'],
                    and data lines added via BTgymMultiData are:
                    {
                        'chf': eurchf_hist_data_source,
                        'gbp', eurgbp_hist_data_source,
                        'jpy', eurgpy_hist_data_source,
                        'usd', eurusd_hist_data_source,
                    },
                    than:

                    env.action.space will be:
                        DictSpace(
                            {
                                'eur': gym.spaces.Box(low=0, high=1, dtype=np.float32),
                                'chf': gym.spaces.Box(low=0, high=1, dtype=np.float32),
                                'gbp': gym.spaces.Box(low=0, high=1, dtype=np.float32),
                                'jpy': gym.spaces.Box(low=0, high=1, dtype=np.float32),
                                'usd': gym.spaces.Box(low=0, high=1, dtype=np.float32),
                            }
                        )

                    single environment action instance (as seen inside strategy or passed to environment via .step()):
                        {
                            'eur': 0.3
                            'chf': 0.1,
                            'gbp': 0.1,
                            'jpy': 0.2,
                            'usd': 0.3,
                        }

                    or vector (unlike multi-asset discrete setup, there is no binary/one hot encoding):
                        (0.3, 0.1, 0.1, 0.2, 0.3)

                    which says to broker: "... adjust positions to get 30% in base EUR asset (cash), and amounts of
                    10%, 10%, 20% and 30% off current portfolio value in CHF, GBP, JPY respectively".

                    Note that under the hood broker uses `order_target_percent` for every risky asset and can issue
                    'sell', 'buy' or 'close' orders depending on positive/negative difference of current to desired
                    share of asset.

            3. Observation space: is nested DictSpace, where 'external' part part of space should hold specifications
            for every data line added (note that cash asset does not have it's own data line).

                Example::

                    if data lines added via BTgymMultiData are:
                        'chf', 'gbp', 'jpy', 'usd';

                    environment observation space can be DictSpace:
                     {
                        'external': DictSpace(
                            {
                                'usd': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                                'gbp': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                                'chf': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                                'jpy': spaces.Box(low=-1000, high=1000, shape=(128, 1, num_features), dtype=np.float32),
                            }
                        ),
                        'raw': spaces.Box(...),
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
            self.log = Logger('BTgymPortfolioShell_{}'.format(self.task), level=self.log_level)

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

        # self.assets = list(self.dataset.assets)

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
        self.cash_name = self.params['strategy']['cash_name']

        self.params['strategy']['initial_action'] = self.get_initial_action()
        self.params['strategy']['initial_portfolio_action'] = self.get_initial_action()

        self.server_actions = {name: self.params['strategy']['portfolio_actions'] for name in self.asset_names}

        try:
            assert set(self.asset_names).issubset(set(self.data_lines_names))

        except AssertionError:
            msg = 'Assets names should be subset of data_lines names, but got: assets: {}, data_lines: {}'.format(
                set(self.asset_names), set(self.data_lines_names)
            )
            self.log.error(msg)
            raise ValueError(msg)

        try:
            assert self.params['strategy']['portfolio_actions'] is None

        except AssertionError:
            self.log.debug(
                'For continious action space strategy.params[`portfolio_actions`] should be `None`, corrected.'
            )
            self.params['strategy']['portfolio_actions'] = None

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
            base_actions=self.params['strategy']['portfolio_actions'],  # None
            assets=list(self.asset_names) + [self.cash_name]
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
        action = {asset: np.asarray([0.0]) for asset in self.asset_names}
        action[self.cash_name] = np.asarray([1.0])
        return action

    def step(self, action):
        """
        Implementation of OpenAI Gym env.step() method.
        Makes a step in the environment.

        Args:
            action:     int or dict, action compatible to env.action_space

        Returns:
            tuple (Observation, Reward, Info, Done)

        """
        # Are you in the list, ready to go and all that?
        if self.action_space.contains(action) \
                and not self._closed \
                and (self.socket is not None) \
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

        # print('step: ', action, action_as_dict)
        env_response = self._comm_with_timeout(
            socket=self.socket,
            message={'action': action}
        )
        if not env_response['status'] in 'ok':
            msg = '.step(): server unreachable with status: <{}>.'.format(env_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        self.env_response = env_response['message']

        return self.env_response
