
import backtrader as bt
import backtrader.indicators as btind

from gym import spaces
from btgym import DictSpace

import numpy as np
from scipy import stats
from collections import namedtuple

from btgym.research.model_based.model.rec import Zscore


NormalisationState = namedtuple('NormalisationState', ['mean', 'variance', 'low_interval', 'up_interval'])


class BaseStrategy6(bt.Strategy):
    """
    Added for gen.6:
        traded asset volatility-based rescaling for all broker statistics and, consequently, reward fn
        self.p.norm_alpha - tracking smoothing decay parameter added
        self.p.target_call  - upper limit arg. is removed
        TODO: auto sizer inference, co-integration coeff. inference

    Controls Environment inner dynamics and backtesting logic. Provides gym'my (State, Action, Reward, Done, Info) data.
    Any State, Reward and Info computation logic can be implemented by subclassing BTgymStrategy and overriding
    get_[mode]_state(), get_reward(), get_info(), is_done() and set_datalines() methods.
    One can always go deeper and override __init__ () and next() methods for desired
    server cerebro engine behaviour, including order execution logic etc.

    Note:
        - base class supports single asset iteration via default data_line named 'base_asset', see derived classes
          multi-asset support
        - bt.observers.DrawDown observer will be automatically added to BTgymStrategy instance at runtime.
        - Since it is bt.Strategy subclass, refer to https://www.backtrader.com/docu/strategy.html for more information.
    """
    # Time embedding period:
    time_dim = 32  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = int(time_dim / 2)

    # Possible agent actions;  Note: place 'hold' first! :
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    features_parameters = ()
    num_features = len(features_parameters)

    params = dict(
        # Observation state shape is dictionary of Gym spaces,
        # at least should contain `raw_state` field.
        # By convention first dimension of every Gym Box space is time embedding one;
        # one can define any shape; should match env.observation_space.shape.
        # observation space state min/max values,
        # For `raw_state' (default) - absolute min/max values from BTgymDataset will be used.
        state_shape={
            'raw': spaces.Box(
                shape=(time_dim, 4),
                low=0,  # will get overridden.
                high=0,
                dtype=np.float32,
            ),
            'internal': spaces.Box(low=-100, high=100, shape=(avg_period, 1, 5), dtype=np.float32),
            'stat': spaces.Box(low=-100, high=100, shape=(2, 1), dtype=np.float32),
            'metadata': DictSpace(
                {
                    'type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'trial_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'trial_type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'sample_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'first_row': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'timestamp': spaces.Box(
                        shape=(),
                        low=0,
                        high=np.finfo(np.float64).max,
                        dtype=np.float64
                    ),
                }
            )
        },
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        slippage=None,
        leverage=1.0,
        gamma=0.99,             # fi_gamma, should match MDP gamma decay
        reward_scale=1.0,       # reward multiplicator
        norm_alpha=0.001,       # renormalisation tracking decay in []0, 1]
        drawdown_call=10,       # finish episode when hitting drawdown treshghold, in percent to initial cash.
        dataset_stat=None,      # Summary descriptive statistics for entire dataset and
        episode_stat=None,      # current episode. Got updated by server.
        time_dim=time_dim,      # time embedding period
        avg_period=avg_period,  # number of time steps reward estimation statistics are tracked over
        features_parameters=features_parameters,
        num_features=num_features,
        metadata={},
        broadcast_message={},
        trial_stat=None,
        trial_metadata=None,
        portfolio_actions=portfolio_actions,
        skip_frame=1,       # number of environment steps to skip before returning next environment response
        order_size=None,
        initial_action=None,
        initial_portfolio_action=None,
        state_int_scale=1,
        state_ext_scale=1,
    )

    def __init__(self, **kwargs):
        """
        Keyword Args:
            params (dict):          parameters dictionary, see Note below.

            Notes:
                Due to backtrader convention, any strategy arguments should be defined inside `params` dictionary
                or passed as kwargs to bt.Cerebro() class via .addstrategy() method. Parameter dictionary
                should contain at least these keys::

                    state_shape:        Observation state shape is dictionary of Gym spaces, by convention
                                        first dimension of every Gym Box space is time embedding one;
                    cash_name:          str, name for cash asset
                    asset_names:        iterable of str, names for assets
                    start_cash:         float, broker starting cash
                    commission:         float, broker commission value, .01 stands for 1%
                    leverage:           float, broker leverage
                    slippage:           float, broker execution slippage
                    order_size:         dict of fixed order stakes (floats); keys should match assets names.
                    drawdown_call:      finish episode when hitting this drawdown treshghold , in percent.
                    portfolio_actions:  possible agent actions.
                    skip_frame:         number of environment steps to skip before returning next response,
                                        e.g. if set to 10 -- agent will interact with environment every 10th step;
                                        every other step agent action is assumed to be 'hold'.

                Default values are::

                    state_shape=dict(raw_state=spaces.Box(shape=(4, 4), low=0, high=0,))
                    cash_name='default_cash'
                    asset_names=['default_asset']
                    start_cash=None
                    commission=None
                    slippage=None,
                    leverage=1.0
                    drawdown_call=10
                    dataset_stat=None
                    episode_stat=None
                    portfolio_actions=('hold', 'buy', 'sell', 'close')
                    skip_frame=1
                    order_size=None
        """
        # Inherit logger from cerebro:
        self.log = self.env._log

        assert self.p.avg_period + 2 < self.p.time_dim, 'Doh!'

        self.skip_frame = self.p.skip_frame

        self.iteration = 0
        self.pre_iteration = 0
        self.env_iteration = 0
        self.inner_embedding = 1
        self.is_done = False
        self.is_done_enabled = False
        self.steps_till_is_done = 2  # extra steps to make when episode terminal conditions are met
        self.action = self.p.initial_portfolio_action
        self.action_to_repeat = self.p.initial_portfolio_action
        self.action_repeated = 0
        self.num_action_repeats = None
        self.reward = 0
        self.order = None
        self.order_failed = 0
        self.broker_message = '_'
        self.final_message = '_'
        self.raw_state = None
        self.time_stamp = 0

        # Prepare broker:
        if self.p.start_cash is not None:
            self.env.broker.setcash(self.p.start_cash)

        if self.p.commission is not None:
            self.env.broker.setcommission(commission=self.p.commission, leverage=self.p.leverage)

        if self.p.slippage is not None:
            # Bid/ask workaround: set overkill 10% slippage + slip_out=False
            # ensuring we always buy at current 'high'~'ask' and sell at 'low'~'bid':
            self.env.broker.set_slippage_perc(self.p.slippage, slip_open=True, slip_match=True, slip_out=False)

        # self.target_value = self.env.broker.startingcash * (1 + self.p.target_call / 100)

        # Try to define stake, if no self.p.order_size dict has been set:
        if self.p.order_size is None:
            # If no order size has been set for every data_line,
            # try to infer stake size from sizer set by bt.Cerebro.addsizer() method:
            try:
                assert len(list(self.env.sizers.values())) == 1
                env_sizer_params = list(self.env.sizers.values())[0][-1]  # pull dict of outer set sizer params
                assert 'stake' in env_sizer_params.keys()

            except (AssertionError, KeyError) as e:
                msg = 'Order stake is not set neither via strategy.param.order_size nor via bt.Cerebro.addsizer method.'
                self.log.error(msg)
                raise ValueError(msg)

            self.p.order_size = {name: env_sizer_params['stake'] for name in self.p.asset_names}

        elif isinstance(self.p.order_size, int) or isinstance(self.p.order_size, float):
            unimodal_stake = {name: self.p.order_size for name in self.getdatanames()}
            self.p.order_size = unimodal_stake

        # Current effective order sizes:
        self.current_order_sizes = None
        self.margin_reserve = 0.01

        # Current stat normalisation:
        self.normalizer = 1.0

        # self.log.warning('asset names: {}'.format(self.p.asset_names))
        # self.log.warning('data names: {}'.format(self.getdatanames()))

        self.trade_just_closed = False
        self.trade_result = 0

        self.unrealized_pnl = None
        self.norm_broker_value = None
        self.realized_pnl = None

        self.current_pos_duration = 0
        self.current_pos_min_value = 0
        self.current_pos_max_value = 0

        self.realized_broker_value = self.env.broker.startingcash
        self.episode_result = 0  # not used

        # Service sma to get correct first features values:
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=self.p.time_dim
        )
        self.data.dim_sma.plotinfo.plot = False

        # self.log.warning('self.p.dir: {}'.format(dir(self.params)))

        # Episode-wide metadata:
        self.metadata = {
            'type': np.asarray(self.p.metadata['type']),
            'trial_num': np.asarray(self.p.metadata['parent_sample_num']),
            'trial_type': np.asarray(self.p.metadata['parent_sample_type']),
            'sample_num': np.asarray(self.p.metadata['sample_num']),
            'first_row': np.asarray(self.p.metadata['first_row']),
            'timestamp': np.asarray(self.time_stamp, dtype=np.float64)
        }
        self.state = {
            'raw': None,
            'metadata': None
        }

        # If it is train or test episode?
        # default logic: true iff. it is test episode from target domain:
        self.is_test = self.metadata['type'] and self.metadata['trial_type']

        # This flag shows to the outer world if this episode can broadcast world-state information, e.g. move global
        # time forward (see: btgym.server._BTgymAnalyzer.next() method);
        self.can_broadcast = self.is_test

        self.log.debug('strategy.metadata: {}'.format(self.metadata))
        self.log.debug('is_test: {}'.format(self.is_test))

        # Broker data lines of interest (used for estimation inner state of agent:
        self.broker_datalines = [
            'cash',
            'value',
            'exposure',
            'drawdown',
            'pos_duration',
            'realized_pnl',
            'unrealized_pnl',
            'min_unrealized_pnl',
            'max_unrealized_pnl',
            'total_unrealized_pnl',
        ]
        # Define flat collection dictionary looking up for methods for estimating broker statistics,
        # one method for one mode, should be named .get_broker_[mode_name]():
        self.collection_get_broker_stat_methods = {}
        for line in self.broker_datalines:
            try:
                self.collection_get_broker_stat_methods[line] = getattr(self, 'get_broker_{}'.format(line))

            except AttributeError:
                raise NotImplementedError('Callable get_broker_{}.() not found'.format(line))

        # Broker and account related sliding statistics accumulators:
        self.broker_stat = {key: np.zeros(self.avg_period) for key in self.broker_datalines}

        # This data line will be used to by default to
        # define normalisation bounds (can be overiden via .set_datalines()):
        self.stat_asset = self.data.open

        # Add custom data Lines if any [and possibly redefine stat_asset and order_size_normalizer]:
        self.set_datalines()

        # Normalisation statistics estimator (updated via update_broker_stat.()):
        self.norm_stat_tracker = Zscore(1, alpha=self.p.norm_alpha)
        self.normalisation_state = NormalisationState(0, 0, .9, 1.1)

        # State exp. smoothing params:
        # self.internal_state_discount = np.cumprod(np.tile(1 - 1 / self.p.avg_period, self.p.avg_period))[::-1]
        # self.external_state_discount = None  # not used

        # Define flat collection dictionary looking for methods for estimating observation state,
        # one method per one mode, should be named .get_[mode_name]_state():
        self.collection_get_state_methods = {}
        for key in self.p.state_shape.keys():
            try:
                self.collection_get_state_methods[key] = getattr(self, 'get_{}_state'.format(key))

            except AttributeError:
                raise NotImplementedError('Callable get_{}_state.() not found'.format(key))

        for data in self.datas:
            self.log.debug('data_name: {}'.format(data._name))

        self.log.debug('stake size: {}'.format(self.p.order_size))

        # Define how this strategy should handle actions: either as discrete or continuous:
        if self.p.portfolio_actions is None or set(self.p.portfolio_actions) == {}:
            # No discrete actions provided, assume continuous:
            try:
                assert self.p.skip_frame > 1

            except AssertionError:
                msg = 'For continuous actions it is essential to set `skip_frame` parameter > 1, got: {}'.format(
                    self.p.skip_frame
                )
                self.log.error(msg)
                raise ValueError(msg)
            # Disable broker checking margin,
            # see: https://community.backtrader.com/topic/152/multi-asset-ranking-and-rebalancing/2?page=1
            self.env.broker.set_checksubmit(False)
            self.next_process_fn = self._next_target_percent
            # Repeat action 2 times:
            self.num_action_repeats = 2

        else:
            # Use discrete handling method otherwise:
            self.env.broker.set_checksubmit(True)
            self.next_process_fn = self._next_discrete
            # self.log.warning('DISCRETE')
            # Do not repeat action for discrete:
            self.num_action_repeats = 0

    def prenext(self):
        if self.pre_iteration + 2 > self.p.time_dim - self.avg_period:
            self.update_broker_stat()

        elif self.pre_iteration + 2 == self.p.time_dim - self.avg_period:
            _ = self.norm_stat_tracker.reset(
                np.asarray(self.stat_asset.get(size=self.data.close.buflen()))[None, :]
            )

        self.pre_iteration += 1

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        # self.log.warning('Inner time embedding: {}'.format(self.inner_embedding))
        # for k, v in self.broker_stat.items():
        #     self.log.warning('{}: {}'.format(k, len(v)))

    def next(self):
        """
        Default implementation for built-in backtrader method.
        Defines one step environment routine;
        Handles order execution logic according to action received.
        Note that orders can only be submitted for data_lines in action_space (assets).
        `self.action` attr. is updated by btgym.server._BTgymAnalyzer, and `None` actions
        are emitted while doing `skip_frame` loop.
        """
        self.update_broker_stat()

        if '_skip_this' in self.action.keys():
            # print('a_skip, b_message: ', self.broker_message)
            if self.action_repeated < self.num_action_repeats:
                self.next_process_fn(self.action_to_repeat)
                self.action_repeated += 1

        else:
            self.next_process_fn(self.action)
            self.action_repeated = 0
            self.action_to_repeat = self.action
            # print('a_process, b_message: ', self.broker_message)

    def notify_trade(self, trade):
        if trade.isclosed:
            # Set trade flags: True if trade have been closed just now and within last frame-skip period,
            # and store trade result:
            self.trade_just_closed = True
            # Note: `trade_just_closed` flag has to be reset manually after evaluating.
            self.trade_result += trade.pnlcomm

            # Store realized prtfolio value:
            self.realized_broker_value = self.broker.get_value()
            # self.log.warning('notify_trade: trade_pnl: {}, cum_trade_result: {}, realized_value: {}'.format(
            #     trade.pnlcomm, self.trade_result, self.realized_broker_value)
            # )

    def update_broker_stat(self):
        """
        Updates all sliding broker statistics with latest-step values such as:
            - normalized broker value
            - normalized broker cash
            - normalized exposure (position size)
            - exp. scaled episode duration in steps, normalized wrt. max possible episode steps
            - normalized realized profit/loss for last closed trade (is zero if no pos. closures within last env. step)
            - normalized profit/loss for current opened trade (unrealized p/l);
        """
        # Update current account value:
        current_value = self.env.broker.get_value()

        # ...normalisation bounds:
        norm_state = self.get_normalisation()

        # ..current order sizes:

        # order_sizes = self.get_order_sizes()


        # ...individual positions for each instrument traded:
        positions = [self.env.broker.getposition(data) for data in self.datas]

        # ... total cash exposure:
        exposure = sum([abs(pos.size) for pos in positions])

        # ... tracking normalisation constant:

        self.normalizer = 1 / np.clip(
            (norm_state.up_interval - norm_state.low_interval),
            1e-8,
            None
        )

        # print('norm_state: ', norm_state)
        # print('normalizer: ', normalizer)
        # print('self.current_order_sizes: ', self.current_order_sizes)

        for key, method in self.collection_get_broker_stat_methods.items():
            update = method(
                current_value=current_value,
                positions=positions,
                exposure=exposure,
                lower_bound=norm_state.low_interval,
                upper_bound=norm_state.up_interval,
                normalizer=self.normalizer,
            )
            # Update accumulator:
            self.broker_stat[key] = np.concatenate([self.broker_stat[key][1:], np.asarray([float(update)])])

        # Reset one-time flags:
        self.trade_just_closed = False
        self.trade_result = 0

    def get_normalisation(self):
        """
        Estimates current normalisation constants, updates `normalisation_state` attr.

        Returns:
            instance of NormalisationState tuple
        """
        # Update normalizer stat:
        stat_data = np.asarray(self.stat_asset.get(size=1))
        mean, var = self.norm_stat_tracker.update(stat_data[None, :])
        var = np.clip(var, 1e-8, None)

        # Use 99% N(stat_data_mean, stat_data_std) intervals as normalisation interval:
        intervals = stats.norm.interval(.99, mean, var ** .5)
        self.normalisation_state = NormalisationState(
            mean=float(mean),
            variance=float(var),
            low_interval=intervals[0][0],
            up_interval=intervals[1][0]
        )
        return self.normalisation_state

    def get_order_sizes(self):
        """
        Estimates current order sizes for assets in trade, sets attribute.

        Returns:
            array-like of floats
        """
        # Default implementation for fixed-size orders:
        self.current_order_sizes = np.fromiter(self.p.order_size.values(), dtype=np.float)
        return self.current_order_sizes

    def get_broker_value(self, current_value, normalizer, **kwargs):
        """

        Args:
            current_value:  float, current portfolio value
            lower_bound:    float, lower normalisation constant
            upper_bound:    float, upper normalisation constant

        Returns:
            broker value normalized w.r.t. start value.
        """
        return (current_value - self.env.broker.startingcash) / self.env.broker.startingcash / self.p.leverage #* normalizer

    def get_broker_cash(self, current_value, **kwargs):
        """
        Args:
            current_value:    float, current portfolio value

        Returns:
            broker cash normalized w.r.t. current value.
        """
        return self.env.broker.get_cash() / current_value

    def get_broker_exposure(self, exposure, normalizer, **kwargs):
        """
        Args:
            exposure:   float, current total position exposure

        Returns:
            exposure (position size) normalized w.r.t. single order size.
        """
        return exposure * normalizer #/ self.current_order_sizes.mean()

    def get_broker_realized_pnl(self, normalizer, **kwargs):
        """

        Args:
            normalizer:     float, normalisation constant

        Returns:
            normalized realized profit/loss for last closed trade (is zero if no pos. closures within last env. step)
        """

        if self.trade_just_closed:
            pnl = self.trade_result * normalizer

        else:
            pnl = 0.0
        return pnl

    def get_broker_unrealized_pnl(self, current_value, normalizer, **kwargs):
        """

        Args:
            current_value:  float, current portfolio value
            normalizer:     float, normalisation constant

        Returns:
            normalized profit/loss for current opened trade
        """
        pnl = (current_value - self.realized_broker_value) * normalizer

        return pnl

    def get_broker_total_unrealized_pnl(self, current_value, normalizer, **kwargs):
        """
        REDUNDANT
        Args:
            current_value:  float, current portfolio value
            normalizer:     float, normalisation constant


        Returns:
            normalized profit/loss wrt. initial portfolio value
        """
        pnl = (current_value - self.env.broker.startingcash) * self.env.broker.startingcash

        return pnl

    def get_broker_drawdown(self, **kwargs):
        """

        Returns:
            current drawdown value
        """
        try:
            dd = self.stats.drawdown.drawdown[-1] / self.p.drawdown_call
        except IndexError:
            dd = 0.0
        return dd

    def get_broker_pos_duration(self, exposure, **kwargs):
        """

        Args:
            exposure:   float, current total positions exposure

        Returns:
            int, number of ticks current position is being held
        """
        if exposure == 0:
            self.current_pos_duration = 0
            # print('ZERO_POSITION\n')

        else:
            self.current_pos_duration += 1

        return self.current_pos_duration

    def get_broker_max_unrealized_pnl(self, current_value, exposure, normalizer, **kwargs):
        """

        Args:
            exposure:       float, current total positions exposure
            current_value:  float, current portfolio value
            normalizer:     float, normalisation constant

        Returns:
            best unrealised PnL achieved within current opened position

        """
        if exposure == 0:
            self.current_pos_max_value = current_value

        else:
            if self.current_pos_max_value < current_value:
                self.current_pos_max_value = current_value

        pnl = (self.current_pos_max_value - self.realized_broker_value) * normalizer

        return pnl

    def get_broker_min_unrealized_pnl(self, current_value, exposure, normalizer, **kwargs):
        """

        Args:
            exposure:       float, current total positions exposure
            current_value:  float, current portfolio value
            normalizer:     float, normalisation constant

        Returns:
            worst unrealised PnL achieved within current opened position
        """
        if exposure == 0:
            self.current_pos_min_value = current_value

        else:
            if self.current_pos_min_value > current_value:
                self.current_pos_min_value = current_value

        pnl = (self.current_pos_min_value - self.realized_broker_value) * normalizer

        return pnl

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close, Volume.
        Any other custom data lines, indicators, etc. should be explicitly defined by overriding this method.
        Invoked once by Strategy.__init__().
        """
        pass

    def get_raw_state(self):
        """
        Default state observation composer.

        Returns:
             and updates time-embedded environment state observation as [n, 4] numpy matrix, where:
                4 - number of signal features  == state_shape[1],
                n - time-embedding length  == state_shape[0] == <set by user>.

        Note:
            `self.raw_state` is used to render environment `human` mode and should not be modified.

        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.high.get(size=self.time_dim)),
                np.frombuffer(self.data.low.get(size=self.time_dim)),
                np.frombuffer(self.data.close.get(size=self.time_dim)),
            )
        ).T

        return self.raw_state

    def get_stat_state(self):
        return np.asarray(self.norm_stat_tracker.get_state())

    def get_internal_state(self):
        stat_lines = ('value', 'unrealized_pnl', 'realized_pnl', 'cash', 'exposure')
        # Use smoothed values:
        x_broker = np.stack(
            [np.asarray(self.broker_stat[name]) for name in stat_lines],
            axis=-1
        )
        # x_broker = np.gradient(x_broker, axis=-1)
        return np.clip(x_broker[:, None, :], -100, 100)

    def get_metadata_state(self):
        self.metadata['timestamp'] = np.asarray(self._get_timestamp())

        return self.metadata

    def _get_time(self):
        """
        Retrieves current time point of the episode data.

        Returns:
            datetime object
        """
        return self.data.datetime.datetime()

    def _get_timestamp(self):
        """
        Sets attr. and returns current data timestamp.

        Returns:
            POSIX timestamp
        """
        self.time_stamp = self._get_time().timestamp()

        return self.time_stamp

    def _get_broadcast_info(self):
        """
        Transmits broadcasting message.

        Returns:
            dictionary  or None
        """
        try:
            return self.get_broadcast_message()

        except AttributeError:
            return None

    def get_broadcast_message(self):
        """
        Override this.

        Returns:
            dictionary or None
        """
        return None

    def get_state(self):
        """
        Collects estimated values for every mode of observation space by calling methods from
        `collection_get_state_methods` dictionary.
        As a rule, this method should not be modified, override or implement corresponding get_[mode]_state() methods,
        defining necessary calculations and return properly shaped tensors for every space mode.

        Note:
            - 'data' referes to bt.startegy datafeeds and should be treated as such.
                Datafeed Lines that are not default to BTgymStrategy should be explicitly defined by
                 __init__() or define_datalines().
        """
        # Update inner state statistic and compose state: <- moved to .next()
        # self.update_broker_stat()
        self.state = {key: method() for key, method in self.collection_get_state_methods.items()}
        return self.state

    def get_reward(self):
        """
        Shapes reward function as normalized single trade realized profit/loss,
        augmented with potential-based reward shaping functions in form of:
        F(s, a, s`) = gamma * FI(s`) - FI(s);
        Potential FI_1 is current normalized unrealized profit/loss.

        Paper:
            "Policy invariance under reward transformations:
             Theory and application to reward shaping" by A. Ng et al., 1999;
             http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
        """

        # All sliding statistics for this step are already updated by get_state().

        # Potential-based shaping function 1:
        # based on potential of averaged profit/loss for current opened trade (unrealized p/l):
        unrealised_pnl = np.asarray(self.broker_stat['unrealized_pnl'])
        current_pos_duration = int(self.broker_stat['pos_duration'][-1])

        #self.log.warning('current_pos_duration: {}'.format(current_pos_duration))

        # We want to estimate potential `fi = gamma*fi_prime - fi` of current opened position,
        # thus need to consider different cases given skip_fame parameter:
        if current_pos_duration == 0:
            # Set potential term to zero if there is no opened positions:
            f1 = 0
            fi_1_prime = 0
        else:
            if current_pos_duration < self.p.skip_frame:
                fi_1 = 0
                fi_1_prime = np.average(unrealised_pnl[-current_pos_duration:])

            elif current_pos_duration < 2 * self.p.skip_frame:
                fi_1 = np.average(
                    unrealised_pnl[-(self.p.skip_frame + current_pos_duration):-self.p.skip_frame]
                )
                fi_1_prime = np.average(unrealised_pnl[-self.p.skip_frame:])

            else:
                fi_1 = np.average(
                    unrealised_pnl[-2 * self.p.skip_frame:-self.p.skip_frame]
                )
                fi_1_prime = np.average(unrealised_pnl[-self.p.skip_frame:])

            # Potential term:
            f1 = self.p.gamma * fi_1_prime - fi_1

        # Main reward function: normalized realized profit/loss:
        realized_pnl = np.asarray(self.broker_stat['realized_pnl'])[-self.p.skip_frame:].sum()

        # Weights are subject to tune:
        self.reward = (0.1 * f1 + 1.0 * realized_pnl) * self.p.reward_scale #/ self.normalizer
        # self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)
        self.reward = np.clip(self.reward, -1e3, 1e3)

        return self.reward

    def get_info(self):
        """
        Composes information part of environment response,
        can be any object. Override to own taste.

        Note:
            Due to 'skip_frame' feature, INFO part of environment response transmitted by server can be  a list
            containing either all skipped frame's info objects, i.e. [info[-9], info[-8], ..., info[0]] or
            just latest one, [info[0]]. This behaviour is set inside btgym.server._BTgymAnalyzer().next() method.
        """
        return dict(
            step=self.iteration,
            time=self.data.datetime.datetime(),
            action=self.action,
            broker_message=self.broker_message,
            broker_cash=self.stats.broker.cash[0],
            broker_value=self.stats.broker.value[0],
            drawdown=self.stats.drawdown.drawdown[0],
            max_drawdown=self.stats.drawdown.maxdrawdown[0],
        )

    def get_done(self):
        """
        Episode termination estimator,
        defines any trading logic conditions episode stop is called upon, e.g. <OMG! Stop it, we became too rich!>.
        It is just a structural a convention method. Default method is empty.

        Expected to return:
            tuple (<is_done, type=bool>, <message, type=str>).
        """
        return False, '-'

    def _get_done(self):
        """
        Default episode termination method,
        checks base conditions episode stop is called upon:
            1. Reached maximum episode duration. Need to check it explicitly, because <self.is_done> flag
               is sent as part of environment response.
            2. Got '_done' signal from outside. E.g. via env.reset() method invoked by outer RL algorithm.
            3. Hit `drawdown` threshold.

        This method shouldn't be overridden or called explicitly.

        Runtime execution logic is:
            terminate episode if:
                get_done() returned (True, 'something')
                OR
                ANY _get_done() default condition is met.
        """
        if not self.is_done_enabled:
            # Episode is on its way,
            # apply base episode termination rules:
            is_done_rules = [
                # Do we approaching the end of the episode?:
                (self.iteration >= \
                 self.data.numrecords - self.inner_embedding - self.p.skip_frame - self.steps_till_is_done,
                 'END OF DATA'),
                # Any money left?:
                (self.stats.drawdown.maxdrawdown[0] >= self.p.drawdown_call, 'DRAWDOWN CALL'),
            ]
            # Append custom get_done() results, if any:
            is_done_rules += [self.get_done()]

            # self.log.debug(
            #     'iteration: {}, condition: {}'.format(
            #         self.iteration,
            #         self.data.numrecords - self.inner_embedding - self.p.skip_frame - self.steps_till_is_done
            #     )
            # )

            # Sweep through rules:
            for (condition, message) in is_done_rules:
                if condition:
                    # Start episode termination countdown for clean exit:
                    # to forcefully execute final `close` order and compute proper reward
                    # we need to make `steps_till_is_done` number of steps until `is_done` flag can be safely risen:
                    self.is_done_enabled = True
                    self.broker_message += message
                    self.final_message = message
                    self.order = self.close()
                    self.log.debug(
                        'Episode countdown started at: {}, {}, r:{}'.format(self.iteration, message, self.reward)
                    )

        else:
            # Now in episode termination phase,
            # just keep hitting `Close` button:
            self.steps_till_is_done -= 1
            self.broker_message = 'CLOSE, {}'.format(self.final_message)
            self.order = self.close()
            self.log.debug(
                'Episode countdown contd. at: {}, {}, r:{}'.format(self.iteration, self.broker_message, self.reward)
            )

        if self.steps_till_is_done <= 0:
            # Now we've done, terminate:
            self.is_done = True

        return self.is_done

    def notify_order(self, order):
        """
        Shamelessly taken from backtrader tutorial.
        TODO: better multi data support
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.broker_message = 'BUY executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

            else:  # Sell
                self.broker_message = 'SELL executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.broker_message = 'ORDER FAILED with status: ' + str(order.getstatusname())
            # Rise order_failed flag until get_reward() will [hopefully] use and reset it:
            self.order_failed += 1
        # self.log.warning('BM: {}'.format(self.broker_message))
        self.order = None

    def _next_discrete(self, action):
        """
        Default implementation for discrete actions.
        Note that orders can be submitted only for data_lines in action_space (assets).

        Args:
            action:     dict, string encoding of btgym.spaces.ActionDictSpace

        """
        for key, single_action in action.items():
            # Simple action-to-order logic:
            if single_action == 'hold' or self.is_done_enabled:
                pass
            elif single_action == 'buy':
                self.order = self.buy(data=key, size=self.p.order_size[key])
                self.broker_message = 'new {}_BUY created; '.format(key) + self.broker_message
            elif single_action == 'sell':
                self.order = self.sell(data=key, size=self.p.order_size[key])
                self.broker_message = 'new {}_SELL created; '.format(key) + self.broker_message
            elif single_action == 'close':
                self.order = self.close(data=key)
                self.broker_message = 'new {}_CLOSE created; '.format(key) + self.broker_message

        # Somewhere after this point, server-side _BTgymAnalyzer() is exchanging information with environment wrapper,
        # obtaining <self.action> , composing and sending <state,reward,done,info> etc... never mind.

    def _next_target_percent(self, action):
        """
        Uses `order_target_percent` method to rebalance assets to given ratios. Expects action for every asset to be
        a float scalar in [0,1], with actions sum to 1 over all assets (including base one).
        Note that action for base asset (cash) is ignored.
        For details refer to: https://www.backtrader.com/docu/order_target/order_target.html
        """
        # TODO 1: filter similar actions to prevent excessive orders issue e.g by DKL on two consecutive ones
        # TODO 2: actions discretisation on level of execution
        for asset in self.p.asset_names:
            # Reducing assets positions subj to 5% margin reserve:
            single_action = round(float(action[asset]) * 0.9, 2)
            self.order = self.order_target_percent(data=asset, target=single_action)
            self.broker_message += ' new {}->{:1.0f}% created; '.format(asset, single_action * 100)

