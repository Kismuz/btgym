import backtrader.indicators as btind

from gym import spaces
from btgym import DictSpace

import numpy as np
from scipy import stats
from pykalman import KalmanFilter

from btgym.research.strategy_gen_6.base import BaseStrategy6, Zscore, NormalisationState
from btgym.research.strategy_gen_6.utils import SpreadSizer, SpreadConstructor, CumSumReward
from btgym.research.model_based.model.bivariate import BivariatePriceModel
from btgym.research.model_based.model.utils import cov2corr, log_stat2stat


class PairSpreadStrategy_0(BaseStrategy6):
    """
    Expects pair of data streams. Forms spread as only virtual trading asset.
    """

    # Time embedding period:
    time_dim = 128  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 64

    # Possible agent actions;  Note: place 'hold' first! :
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    features_parameters = (1, 4, 16, 64, 256, 1024)
    num_features = len(features_parameters)

    params = dict(
        state_shape={
            'external': spaces.Box(low=-10, high=10, shape=(time_dim, 1, num_features*2), dtype=np.float32),
            'internal': spaces.Box(low=-100, high=100, shape=(avg_period, 1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),
            'stat': spaces.Box(low=-100, high=100, shape=(3, 1), dtype=np.float32),
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
                        high=10**10,
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
                        high=10**10,
                        dtype=np.uint32
                    ),
                    'first_row': spaces.Box(
                        shape=(),
                        low=0,
                        high=10**10,
                        dtype=np.uint32
                    ),
                    'timestamp': spaces.Box(
                        shape=(),
                        low=0,
                        high=np.finfo(np.float64).max,
                        dtype=np.float64
                    ),
                    # TODO: make generator parameters names standard
                    'generator': DictSpace(
                        {
                            'mu': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'l': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'sigma': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'x0': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            )
                        }
                    )
                }
            )
        },
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        slippage=None,
        leverage=1.0,
        gamma=1.0,              # fi_gamma, ~ should match MDP gamma decay
        reward_scale=1,         # reward multiplicator
        norm_alpha=0.001,       # renormalisation tracking decay in []0, 1]
        norm_alpha_2=0.01,      # float in []0, 1], tracking decay for original prices
        drawdown_call=10,       # finish episode when hitting drawdown treshghold, in percent.
        dataset_stat=None,      # Summary descriptive statistics for entire dataset and
        episode_stat=None,      # current episode. Got updated by server.
        time_dim=time_dim,      # time embedding period
        avg_period=avg_period,  # number of time steps reward estimation statistics are averaged over
        features_parameters=features_parameters,
        num_features=num_features,
        metadata={},
        broadcast_message={},
        trial_stat=None,
        trial_metadata=None,
        portfolio_actions=portfolio_actions,
        skip_frame=1,       # number of environment steps to skip before returning next environment response
        position_max_depth=1,
        order_size=1,  # legacy plug, to be removed <-- rework gen_6.__init__
        initial_action=None,
        initial_portfolio_action=None,
        state_int_scale=1,
        state_ext_scale=1,
    )

    def __init__(self, **kwargs):
        super(PairSpreadStrategy_0, self).__init__(**kwargs)

        assert len(self.p.asset_names) == 1, 'Only one derivative spread asset is supported'
        assert len(self.getdatanames()) == 2, \
            'Expected exactly two input datalines but {} where given'.format(self.getdatanames())

        if isinstance(self.p.asset_names, str):
            self.p.asset_names = [self.p.asset_names]
        self.action_key = list(self.p.asset_names)[0]

        self.current_expert_action = np.zeros(len(self.p.portfolio_actions))
        self.state['metadata'] = self.metadata

        # Infer OU generator params:
        generator_keys = self.p.state_shape['metadata'].spaces['generator'].spaces.keys()
        if 'generator' not in self.p.metadata.keys() or self.p.metadata['generator'] == {}:
            self.metadata['generator'] = {key: np.asarray(0) for key in generator_keys}

        else:
            # self.metadata['generator'] = {key: self.p.metadata['generator'][key] for key in generator_keys}

            # TODO: clean up this mess, refine names:

            self.metadata['generator'] = {
                'l': self.p.metadata['generator']['ou_lambda'],
                'mu': self.p.metadata['generator']['ou_mu'],
                'sigma': self.p.metadata['generator']['ou_sigma'],
                'x0': 0,
            }

            # Make scalars np arrays to comply gym.spaces.Box specs:
            for k, v in self.metadata['generator'].items():
                self.metadata['generator'][k] = np.asarray(v)

        self.last_delta_total_pnl = 0
        self.last_pnl = 0

        self.log.debug('startegy got broadcast_msg: <<{}>>'.format(self.p.broadcast_message))

        # Track original prices statistics, let base self.norm_stat_tracker track spread (=stat_asset) itself:
        self.norm_stat_tracker_2 = Zscore(2, self.p.norm_alpha_2)

        # Synthetic spread order size estimator:
        self.spread_sizer = SpreadSizer(
            init_cash=self.p.start_cash,
            position_max_depth=self.p.position_max_depth,
            leverage=self.p.leverage,
            margin_reserve=self.margin_reserve,
        )
        self.last_action = None

        # Keeps track of virtual spread position
        # long_ spread: >0, short_spread: <0, no positions: 0
        self.spread_position_size = 0

        # Reward signal filtering:
        self.kf = KalmanFilter(
            initial_state_mean=0,
            transition_covariance=.01,
            observation_covariance=1,
            n_dim_obs=1
        )
        self.kf_state = [0, 0]

    def set_datalines(self):
        # Override stat line:
        self.stat_asset = self.data.spread = SpreadConstructor()

        # Spy on reward behaviour:
        self.reward_tracker = CumSumReward()

        self.data.std = btind.StdDev(self.data.spread, period=self.p.time_dim, safepow=True)
        self.data.std.plotinfo.plot = False

        self.data.features = [btind.EMA(self.data.spread, period=period) for period in self.p.features_parameters]
        initial_time_period = np.asarray(self.p.features_parameters).max() + self.p.time_dim
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=initial_time_period
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_broadcast_message(self):
        """
        Not used.
        """
        return {
            'data_model_psi': np.zeros([2, 3]),
            'iteration': self.iteration
        }

    def get_expert_state(self):
        """
        Not used.
        """
        return np.zeros(len(self.p.portfolio_actions))

    def prenext(self):
        if self.pre_iteration + 2 > self.p.time_dim - self.avg_period:
            self.update_broker_stat()
            x_upd = np.stack(
                [
                    np.asarray(self.datas[0].get(size=1)),
                    np.asarray(self.datas[1].get(size=1))
                ],
                axis=0
            )
            _ = self.norm_stat_tracker_2.update(x_upd)  # doubles update_broker_stat() but helps faster stabilization

        elif self.pre_iteration + 2 == self.p.time_dim - self.avg_period:
            # Initialize all trackers:
            x_init = np.stack(
                [
                    np.asarray(self.datas[0].get(size=self.data.close.buflen())),
                    np.asarray(self.datas[1].get(size=self.data.close.buflen()))
                ],
                axis=0
            )
            _ = self.norm_stat_tracker_2.reset(x_init)
            _ = self.norm_stat_tracker.reset(np.asarray(self.stat_asset.get(size=self.data.close.buflen()))[None, :])
            # _ = self.norm_stat_tracker.reset(np.asarray(self.stat_asset.get(size=1))[None, :])

        self.pre_iteration += 1

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

    def get_normalisation(self):
        """
        Estimates current normalisation constants, updates `normalisation_state` attr.

        Returns:
            instance of NormalisationState tuple
        """
        # Update synth. spread rolling normalizers:
        x_upd = np.stack(
            [
                np.asarray(self.datas[0].get(size=1)),
                np.asarray(self.datas[1].get(size=1))
            ],
            axis=0
        )
        _ = self.norm_stat_tracker_2.update(x_upd)

        # ...and use [normalised] spread rolling mean and variance to estimate NormalisationState
        # used to normalize all broker statistics and reward:
        spread_data = np.asarray(self.stat_asset.get(size=1))

        mean, var = self.norm_stat_tracker.update(spread_data[None, :])
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

    def get_stat_state(self):
        return np.concatenate(
            [np.asarray(self.norm_stat_tracker.get_state()), np.asarray(self.stat_asset.get())[None, :]],
            axis=0
        )

    def get_external_state(self):
        """
        Attempt to include avg decomp. of original normalised spread
        """
        x_sma = np.stack(
            [
                feature.get(size=self.p.time_dim) for feature in self.data.features
            ],
            axis=-1
        )
        scale = 1 / np.clip(self.data.std[0], 1e-10, None)
        x_sma *= scale  # <-- more or less ok

        # Gradient along features axis:
        dx = np.gradient(x_sma, axis=-1)

        # TODO: different conv. encoders for these two types of features:
        x = np.concatenate([x_sma, dx], axis=-1)

        # Crop outliers:
        x = np.clip(x, -10, 10)
        return x[:, None, :]

    def get_order_sizes(self):
        """
        Estimates current order sizes for assets in trade, updates attribute.

        Returns:
            array-like of floats
        """
        s = self.norm_stat_tracker_2.get_state()
        self.current_order_sizes = np.asarray(
            self.spread_sizer.get_sizing(self.env.broker.get_value(), s.mean, s.variance),
            dtype=np.float
        )
        return self.current_order_sizes

    def long_spread(self):
        """
        Opens or adds up long spread `virtual position`.
        """
        # Get current sizes:
        order_sizes = self.get_order_sizes()

        if self.spread_position_size >= 0:
            if not self.can_add_up(order_sizes[0], order_sizes[1]):
                self.order_failed += 1
                # self.log.warning(
                #     'Adding Long spread to existing {} hit margin, ignored'.format(self.spread_position_size)
                # )
                return

        elif self.spread_position_size == -1:
            # Currently in single short -> just close to prevent disballance:
            return self.close_spread()

        name1 = self.datas[0]._name
        name2 = self.datas[1]._name

        self.order = self.buy(data=name1, size=order_sizes[0])
        self.order = self.sell(data=name2, size=order_sizes[1])
        self.spread_position_size += 1
        # self.log.warning('long spread submitted, new pos. size: {}'.format(self.spread_position_size))

    def short_spread(self):
        order_sizes = self.get_order_sizes()

        if self.spread_position_size <= 0:
            if not self.can_add_up(order_sizes[0], order_sizes[1]):
                self.order_failed += 1
                # self.log.warning(
                #     'Adding Short spread to existing {} hit margin, ignored'.format(self.spread_position_size)
                # )
                return

        elif self.spread_position_size == 1:
            # Currently in single long:
            return self.close_spread()

        name1 = self.datas[0]._name
        name2 = self.datas[1]._name

        self.order = self.sell(data=name1, size=order_sizes[0])
        self.order = self.buy(data=name2, size=order_sizes[1])
        self.spread_position_size -= 1
        # self.log.warning('short spread submitted, new pos. size: {}'.format(self.spread_position_size))

    def close_spread(self):
        self.order = self.close(data=self.datas[0]._name)
        self.order = self.close(data=self.datas[1]._name)
        self.spread_position_size = 0
        # self.log.warning('close spread submitted, new pos. size: {}'.format(self.spread_position_size))

    def can_add_up(self, order_0_size=None, order_1_size=None):
        """
        Checks if there enough cash left to open synthetic spread position

        Args:
            order_0_size:   float, order size for data0 asset or None
            order_1_size:   float, order size for data1 asset or None

        Returns:
            True if possible, False otherwise
        """
        if order_1_size is None or order_0_size is None:
            order_sizes = self.get_order_sizes()
            order_0_size = order_sizes[0]
            order_1_size = order_sizes[1]

        # Get full operation cost:
        # TODO: it can be two commissions schemes
        op_cost = [
            self.env.broker.comminfo[None].getoperationcost(
                size=size,
                price=self.getdatabyname(name).high[0]
            ) / self.env.broker.comminfo[None].get_leverage() +
            self.env.broker.comminfo[None].getcommission(
                size=size,
                price=self.getdatabyname(name).high[0]
            )
            for size, name in zip([order_0_size, order_1_size], [self.datas[0]._name, self.datas[1]._name])
        ]
        # self.log.warning('op_cost+comm+reserve: {:.4f}'.format(np.asarray(op_cost).sum() + self.margin_reserve))
        # self.log.warning('order sizes: {:.4f}; {:.4f}'.format(order_0_size, order_1_size))
        # self.log.warning('leverage: {}'.format(self.env.broker.comminfo[None].get_leverage()))
        # self.log.warning(
        #     'commision: {:.4f} + {:.4f}'.format(
        #         self.env.broker.comminfo[None].getcommission(
        #             size=order_0_size,
        #             price=self.getdatabyname(self.datas[0]._name).high[0]
        #         ),
        #         self.env.broker.comminfo[None].getcommission(
        #             size=order_1_size,
        #             price=self.getdatabyname(self.datas[1]._name).high[0]
        #         ),
        #     )
        # )
        # self.log.warning('current_cash: {}'.format(self.env.broker.get_cash()))
        if np.asarray(op_cost).sum() + self.margin_reserve >= self.env.broker.get_cash() * (1 - self.margin_reserve):
            # self.log.warning('add_up check failed')
            return False

        else:
            # self.log.warning('add_up check ok')
            return True

    def get_broker_pos_duration(self, **kwargs):
        """
        Position duration is measured w.r.t. virtual spread position, not broker account exposure
        """
        if self.spread_position_size == 0:
            self.current_pos_duration = 0
            # self.log.warning('zero position')

        else:
            self.current_pos_duration += 1
            # self.log.warning('position duration: {}'.format(self.current_pos_duration))

        return self.current_pos_duration

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

        # self.log.warning('BM: {}'.format(self.broker_message))
        self.order = None

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
        current_pos_duration = self.broker_stat['pos_duration'][-1]

        # We want to estimate potential `fi = gamma*fi_prime - fi` of current opened position,
        # thus need to consider different cases given skip_fame parameter:
        if current_pos_duration == 0:
            # Set potential term to zero if there is no opened positions:
            fi_1 = 0
            fi_1_prime = 0
            # Reset filter state:
            self.kf_state = [0, 0]
        else:
            fi_1 = self.last_pnl
            # fi_1_prime = np.average(unrealised_pnl[-1])
            self.kf_state = self.kf.filter_update(
                filtered_state_mean=self.kf_state[0],
                filtered_state_covariance=self.kf_state[1],
                observation=unrealised_pnl[-1],
            )
            fi_1_prime = np.squeeze(self.kf_state[0])

        # Potential term 1:
        f1 = self.p.gamma * fi_1_prime - fi_1
        self.last_pnl = fi_1_prime

        # Potential-based shaping function 2:
        # based on potential of averaged profit/loss for global unrealized pnl:
        total_pnl = np.asarray(self.broker_stat['total_unrealized_pnl'])
        delta_total_pnl = np.average(total_pnl[-self.p.skip_frame:]) - np.average(total_pnl[:-self.p.skip_frame])

        fi_2 = delta_total_pnl
        fi_2_prime = self.last_delta_total_pnl

        # Potential term 2:
        f2 = self.p.gamma * fi_2_prime - fi_2
        self.last_delta_total_pnl = delta_total_pnl

        # Potential term 3:
        # f3 = 1 + 0.5 * np.log(1 + current_pos_duration)
        f3 = 1.0

        # Main reward function: normalized realized profit/loss:
        realized_pnl = np.asarray(self.broker_stat['realized_pnl'])[-self.p.skip_frame:].sum()

        # Weights are subject to tune:
        self.reward = (0.1 * f1 * f3 + 1.0 * realized_pnl) * self.p.reward_scale #/ self.normalizer
        # self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)

        self.reward = np.clip(self.reward, -1e3, 1e3)

        return self.reward

    def _next_discrete(self, action):
        """
        Manages spread virtual positions.

        Args:
            action:     dict, string encoding of btgym.spaces.ActionDictSpace

        """
        # Here we expect action dict to contain single key:
        single_action = action[self.action_key]

        if single_action == 'hold' or self.is_done_enabled:
            pass
        elif single_action == 'buy':
            self.long_spread()
            self.broker_message = 'new {}_LONG created; '.format(self.action_key) + self.broker_message
        elif single_action == 'sell':
            self.short_spread()
            self.broker_message = 'new {}_SHORT created; '.format(self.action_key) + self.broker_message
        elif single_action == 'close':
            self.close_spread()
            self.broker_message = 'new {}_CLOSE created; '.format(self.action_key) + self.broker_message


class PairSpreadStrategy_1(PairSpreadStrategy_0):
    """
    Expects pair of data streams. Encodes each asset independently.
    """

    # Time embedding period:
    time_dim = 128  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 64

    # Possible agent actions;  Note: place 'hold' first! :
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    features_parameters = (1, 4, 16, 64, 256, 1024)
    num_features = len(features_parameters)

    params = dict(
        state_shape={
            'external': DictSpace(
                {
                    'asset1': spaces.Box(low=-10, high=10, shape=(time_dim, 1, num_features), dtype=np.float32),
                    'asset2': spaces.Box(low=-10, high=10, shape=(time_dim, 1, num_features), dtype=np.float32),
                }
            ),

            'internal': spaces.Box(low=-100, high=100, shape=(avg_period, 1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),
            'stat': spaces.Box(low=-100, high=100, shape=(3, 1), dtype=np.float32),
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
                    # TODO: make generator parameters names standard
                    'generator': DictSpace(
                        {
                            'mu': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'l': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'sigma': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'x0': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            )
                        }
                    )
                }
            )
        },
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        slippage=None,
        leverage=1.0,
        gamma=1.0,  # fi_gamma, ~ should match MDP gamma decay
        reward_scale=1,  # reward multiplicator
        norm_alpha=0.001,  # renormalisation tracking decay in []0, 1]
        norm_alpha_2=0.01,  # float in []0, 1], tracking decay for original prices
        drawdown_call=10,  # finish episode when hitting drawdown treshghold, in percent.
        dataset_stat=None,  # Summary descriptive statistics for entire dataset and
        episode_stat=None,  # current episode. Got updated by server.
        time_dim=time_dim,  # time embedding period
        avg_period=avg_period,  # number of time steps reward estimation statistics are averaged over
        features_parameters=features_parameters,
        num_features=num_features,
        metadata={},
        broadcast_message={},
        trial_stat=None,
        trial_metadata=None,
        portfolio_actions=portfolio_actions,
        skip_frame=1,  # number of environment steps to skip before returning next environment response
        position_max_depth=1,
        order_size=1,  # legacy plug, to be removed <-- rework gen_6.__init__
        initial_action=None,
        initial_portfolio_action=None,
        state_int_scale=1,
        state_ext_scale=1,
    )

    def set_datalines(self):
        # Override stat line:
        self.stat_asset = self.data.spread = SpreadConstructor()

        # Spy on reward behaviour:
        self.reward_tracker = CumSumReward()

        self.data.std1 = btind.StdDev(self.datas[0], period=self.p.time_dim, safepow=True)
        self.data.std1.plotinfo.plot = False

        self.data.std2 = btind.StdDev(self.datas[1], period=self.p.time_dim, safepow=True)
        self.data.std2.plotinfo.plot = False

        self.data.features1 = [btind.EMA(self.datas[0], period=period) for period in self.p.features_parameters]
        self.data.features2 = [btind.EMA(self.datas[1], period=period) for period in self.p.features_parameters]

        initial_time_period = np.asarray(self.p.features_parameters).max() + self.p.time_dim
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=initial_time_period
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_external_state(self):
        """
        Attempt to include avg decomp. of original normalised spread
        """
        x_sma1 = np.stack(
            [
                feature.get(size=self.p.time_dim) for feature in self.data.features1
            ],
            axis=-1
        )
        scale = 1 / np.clip(self.data.std1[0], 1e-10, None)
        x_sma1 *= scale  # <-- more or less ok

        # Gradient along features axis:
        dx1 = np.gradient(x_sma1, axis=-1)
        dx1 = np.clip(dx1, -10, 10)

        x_sma2 = np.stack(
            [
                feature.get(size=self.p.time_dim) for feature in self.data.features2
            ],
            axis=-1
        )
        scale = 1 / np.clip(self.data.std2[0], 1e-10, None)
        x_sma2 *= scale  # <-- more or less ok

        # Gradient along features axis:
        dx2 = np.gradient(x_sma2, axis=-1)
        dx2 = np.clip(dx2, -10, 10)

        return {'asset1': dx1[:, None, :], 'asset2': dx2[:, None, :],}


class SSAStrategy_0(PairSpreadStrategy_0):
    """
    BivariateTimeSeriesModel decomposition based.
    """
    time_dim = 128
    avg_period = 16
    model_time_dim = 16
    portfolio_actions = ('hold', 'buy', 'sell', 'close')
    features_parameters = None
    num_features = 4

    params = dict(
        state_shape={
            'external': DictSpace(
                {
                    'ssa': spaces.Box(low=-100, high=100, shape=(time_dim, 1, num_features), dtype=np.float32),

                }
            ),
            'internal': DictSpace(
                {
                    'broker': spaces.Box(low=-100, high=100, shape=(avg_period, 1, 5), dtype=np.float32),
                    'model': spaces.Box(low=-100, high=100, shape=(model_time_dim, 1, 9), dtype=np.float32),
                }
            ),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),  # not used
            'stat': spaces.Box(low=-1e6, high=1e6, shape=(3, 1), dtype=np.float32),  # for debug. proposes only
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
                    'generator': DictSpace(  # ~ S-generator params.
                        {
                            'mu': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'l': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'sigma': spaces.Box(
                                shape=(),
                                low=0,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            ),
                            'x0': spaces.Box(
                                shape=(),
                                low=np.finfo(np.float64).min,
                                high=np.finfo(np.float64).max,
                                dtype=np.float64
                            )
                        }
                    )
                }
            )
        },
        data_model_params=dict(
            alpha=.001,
            stat_alpha=.0001,
            filter_alpha=.05,
            max_length=time_dim * 2,
            analyzer_window=10,
            p_analyzer_grouping=[[0, 1], [1, 2], [2, 3], [3, None]],
            s_analyzer_grouping=[[0, 1], [1, 2], [2, 3], [3, None]]
        ),
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        slippage=None,
        leverage=1.0,
        gamma=1.0,              # fi_gamma, should match MDP gamma decay
        reward_scale=1,         # reward multiplicator
        norm_alpha=0.001,       # float in []0, 1], renormalisation tracking decay (for synth. spread)
        norm_alpha_2=0.01,     # float in []0, 1], tracking decay for original prices
        drawdown_call=10,       # finish episode when hitting drawdown threshold, in percent.
        dataset_stat=None,      # Summary descriptive statistics for entire dataset and
        episode_stat=None,      # current episode. Got updated by server.
        time_dim=time_dim,      # time embedding period
        avg_period=avg_period,  # number of time steps reward estimation statistics are averaged over
        features_parameters=features_parameters,
        num_features=num_features,
        metadata={},
        broadcast_message={},
        trial_stat=None,
        trial_metadata=None,
        portfolio_actions=portfolio_actions,
        skip_frame=1,  # number of environment steps to skip before returning next environment response
        position_max_depth=1,
        order_size=1,  # legacy plug, to be removed <-- rework gen_6.__init__
        initial_action=None,
        initial_portfolio_action=None,
        state_int_scale=1,
        state_ext_scale=1,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Bivariate model:
        self.data_model = BivariatePriceModel(**self.p.data_model_params)

        # Accumulators for 'model' observation mode:
        self.external_model_state = np.zeros([self.model_time_dim, 1, 9])


    def set_datalines(self):
        # Discard superclass dataline, use SpreadConstructor instead:
        self.data.spread = None

        # Override stat line:
        self.stat_asset = self.SpreadConstructor()

        # Spy on reward behaviour:
        self.reward_tracker = self.CumSumReward()

        initial_time_period = self.p.time_dim
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=initial_time_period
        )
        self.data.dim_sma.plotinfo.plot = False

    def prenext(self):
        if self.pre_iteration + 2 > self.p.time_dim - self.avg_period:
            self.update_broker_stat()
            x_upd = np.stack(
                [
                    np.asarray(self.datas[0].get(size=1)),
                    np.asarray(self.datas[1].get(size=1))
                ],
                axis=0
            )
            _ = self.norm_stat_tracker_2.update(x_upd)  # doubles update_broker_stat() but helps faster stabilization
            self.data_model.update(x_upd)

        elif self.pre_iteration + 2 == self.p.time_dim - self.avg_period:
            # Initialize all trackers:
            x_init = np.stack(
                [
                    np.asarray(self.datas[0].get(size=self.data.close.buflen())),
                    np.asarray(self.datas[1].get(size=self.data.close.buflen()))
                ],
                axis=0
            )
            _ = self.norm_stat_tracker_2.reset(x_init)
            _ = self.norm_stat_tracker.reset(np.asarray(self.stat_asset.get(size=self.data.close.buflen()))[None, :])
            # _ = self.norm_stat_tracker.reset(np.asarray(self.stat_asset.get(size=1))[None, :])
            self.data_model.reset(x_init)

        self.pre_iteration += 1

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

        # self.log.warning(
        #     'Pos. max. depth: {}, leverage: {}, order sizes: {:.4f}, {:.4f}'.format(
        #         self.p.position_max_depth,
        #         self.p.leverage,
        #         size_0,
        #         size_1
        #     )
        # )

    def get_normalisation(self):
        """
        Estimates current normalisation constants, updates `normalisation_state` attr.

        Returns:
            instance of NormalisationState tuple
        """
        # Update synth. spread rolling normalizers:
        x_upd = np.stack(
            [
                np.asarray(self.datas[0].get(size=1)),
                np.asarray(self.datas[1].get(size=1))
            ],
            axis=0
        )
        _ = self.norm_stat_tracker_2.update(x_upd)

        # ...and use [normalised] spread rolling mean and variance to estimate NormalisationState
        # used to normalize all broker statistics and reward:
        spread_data = np.asarray(self.stat_asset.get(size=1))

        mean, var = self.norm_stat_tracker.update(spread_data[None, :])
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

    def get_external_state(self):
        return dict(
            ssa=self.get_external_ssa_state(),
        )

    def get_internal_state(self):
        return dict(
            broker=self.get_internal_broker_state(),
            model=self.get_data_model_state(),
        )

    def get_external_ssa_state(self):
        """
        Spread SSA decomposition.
        """
        x_upd = np.stack(
            [
                np.asarray(self.datas[0].get(size=self.p.skip_frame)),
                np.asarray(self.datas[1].get(size=self.p.skip_frame))
            ],
            axis=0
        )
        # self.log.warning('x_upd: {}'.format(x_upd.shape))
        self.data_model.update(x_upd)

        x_ssa = self.data_model.s.transform(size=self.p.time_dim).T  #* self.normalizer

        # Gradient along features axis:
        # dx = np.gradient(x_ssa, axis=-1)
        #
        # # Add up: gradient  along time axis:
        # # dx2 = np.gradient(dx, axis=0)
        #
        # x = np.concatenate([x_ssa_bank, dx], axis=-1)

        # Crop outliers:
        x_ssa = np.clip(x_ssa, -10, 10)
        # x_ssa = np.clip(dx, -10, 10)
        return x_ssa[:, None, :]

    def get_data_model_state(self):
        """
         Spread stochastic model parameters.
        """
        state = self.data_model.s.process.get_state()
        cross_corr = cov2corr(state.filtered.covariance)[[0, 0, 1], [1, 2, 2]]
        update = np.concatenate(
            [
                state.filtered.mean.flatten(),
                state.filtered.variance.flatten(),
                cross_corr,
            ]
        )
        self.external_model_state = np.concatenate(
            [
                self.external_model_state[1:, :, :],
                update[None, None, :]
            ],
            axis=0
        )
        # self.external_model_state = np.gradient(self.external_model_state, axis=-1)
        return self.external_model_state

    def get_internal_broker_state(self):
        stat_lines = ('value', 'unrealized_pnl', 'realized_pnl', 'cash', 'exposure')
        x_broker = np.stack(
            [np.asarray(self.broker_stat[name]) for name in stat_lines],
            axis=-1
        )
        # self.log.warning('broker: {}'.format(x_broker))
        # self.log.warning('Ns: {}'.format(self.normalisation_state))
        # x_broker = np.gradient(x_broker, axis=-1)
        return np.clip(x_broker[:, None, :], -100, 100)





