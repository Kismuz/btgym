import backtrader as bt
import backtrader.indicators as btind

from gym import spaces
from btgym import DictSpace

import numpy as np

from btgym.research.strategy_gen_5.base import BaseStrategy5
from btgym.strategy.utils import tanh


class MonoSpreadOUStrategy_0(BaseStrategy5):
    """
    Expects spread as single generated data stream.
    """
    # Time embedding period:
    time_dim = 128  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 30

    # Possible agent actions;  Note: place 'hold' first! :
    #portfolio_actions = ('hold', 'buy', 'sell', 'close')

    portfolio_actions = ('close', 'buy', 'sell', )

    features_parameters = (1, 4, 16, 64, 256, 1024)
    num_features = len(features_parameters)

    params = dict(
        state_shape={
            'external': spaces.Box(low=-10, high=10, shape=(time_dim, 1, num_features*2), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),
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
        gamma=0.99,             # fi_gamma, should match MDP gamma decay
        reward_scale=1,         # reward multiplicator
        drawdown_call=10,       # finish episode when hitting drawdown treshghold , in percent.
        target_call=10,         # finish episode when reaching profit target, in percent.
        dataset_stat=None,      # Summary descriptive statistics for entire dataset and
        episode_stat=None,      # current episode. Got updated by server.
        time_dim=time_dim,      # time embedding period
        avg_period=avg_period,  # number of time steps reward estimation statistics are averaged over
        features_parameters=features_parameters,
        num_features=num_features,
        metadata={},
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
        super(MonoSpreadOUStrategy_0, self).__init__(**kwargs)
        self.data.high = self.data.low = self.data.close = self.data.open
        self.current_expert_action = np.zeros(len(self.p.portfolio_actions))
        self.state['metadata'] = self.metadata

        # Combined dataset related, infer OU generator params:
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

    def set_datalines(self):
        self.data.high = self.data.low = self.data.close = self.data.open

        self.data.std = btind.StdDev(self.data.open, period=self.p.time_dim, safepow=True)
        self.data.std.plotinfo.plot = False

        self.data.features = [
            btind.SimpleMovingAverage(self.data.open, period=period) for period in self.p.features_parameters
        ]
        initial_time_period = np.asarray(self.p.features_parameters).max() + self.p.time_dim
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=initial_time_period
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_external_state(self):
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

        # Add up: gradient  along time axis:
        dx2 = np.gradient(dx, axis=0)

        # TODO: different conv. encoders for these two types of features:
        x = np.concatenate([dx, dx2], axis=-1)

        # Crop outliers:
        x = np.clip(x, -10, 10)
        return x[:, None, :]

    def get_internal_state(self):

        x_broker = np.concatenate(
            [
                np.asarray(self.broker_stat['value'])[..., None],
                np.asarray(self.broker_stat['unrealized_pnl'])[..., None],
                np.asarray(self.broker_stat['realized_pnl'])[..., None],
                np.asarray(self.broker_stat['cash'])[..., None],
                np.asarray(self.broker_stat['exposure'])[..., None],
            ],
            axis=-1
        )
        x_broker = tanh(np.gradient(x_broker, axis=-1) * self.p.state_int_scale)
        return x_broker[:, None, :]

    def get_expert_state(self):
        """
        Not used.
        """
        return np.zeros(len(self.p.portfolio_actions))


class PairSpreadStrategy_0(MonoSpreadOUStrategy_0):
    """
    Expects pair of data streams. Forms spread as trading asset.
    """
    def __init__(self, **kwargs):
        super(PairSpreadStrategy_0, self).__init__(**kwargs)

        assert len(self.p.asset_names) == 1, 'Only one derivative spread asset is supported'
        if isinstance(self.p.asset_names, str):
            self.p.asset_names = [self.p.asset_names]
        self.action_key = list(self.p.asset_names)[0]

        self.last_action = None

        assert len(self.getdatanames()) == 2, \
            'Expected exactly two input datalines but {} where given'.format(self.getdatanames())

    def set_datalines(self):

        self.data.spread = btind.SimpleMovingAverage(self.datas[0] - self.datas[1], period=1)
        self.data.spread.plotinfo.subplot = True
        self.data.spread.plotinfo.plotabove = True
        self.data.spread.plotinfo.plotname = list(self.p.asset_names)[0]

        self.data.std = btind.StdDev(self.data.spread, period=self.p.time_dim, safepow=True)
        self.data.std.plotinfo.plot = False

        self.data.features = [
            btind.SimpleMovingAverage(self.data.spread, period=period) for period in self.p.features_parameters
        ]
        initial_time_period = np.asarray(self.p.features_parameters).max() + self.p.time_dim
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=initial_time_period
        )
        self.data.dim_sma.plotinfo.plot = False

    def buy_spread(self):
        """
        Opens long spread `virtual position`,
        sized 2x minimum single stake_size
        """
        name1 = self.datas[0]._name
        name2 = self.datas[1]._name
        self.order = self.buy(data=name1, size=self.p.order_size[name1])
        self.order = self.sell(data=name2, size=self.p.order_size[name2])

    def sell_spread(self):
        name1 = self.datas[0]._name
        name2 = self.datas[1]._name
        self.order = self.sell(data=name1, size=self.p.order_size[name1])
        self.order = self.buy(data=name2, size=self.p.order_size[name2])

    def close_spread(self):
        self.order = self.close(data=self.datas[0]._name)
        self.order = self.close(data=self.datas[1]._name)

    def ____next_discrete(self, action):
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
            self.buy_spread()
            self.broker_message = 'new {}_BUY created; '.format(self.action_key) + self.broker_message
        elif single_action == 'sell':
            self.sell_spread()
            self.broker_message = 'new {}_SELL created; '.format(self.action_key) + self.broker_message
        elif single_action == 'close':
            self.close_spread()
            self.broker_message = 'new {}_CLOSE created; '.format(self.action_key) + self.broker_message

    def _next_discrete(self, action):
        """
        Manages spread virtual positions.

        Args:
            action:     dict, string encoding of btgym.spaces.ActionDictSpace

        """
        # Here we expect action dict to contain single key:
        single_action = action[self.action_key]

        if single_action == self.last_action or self.is_done_enabled:
            pass
        elif single_action == 'buy':
            self.buy_spread()
            self.broker_message = '{}_LONG created; '.format(self.action_key) + self.broker_message
        elif single_action == 'sell':
            self.sell_spread()
            self.broker_message = 'new {}_SHORT created; '.format(self.action_key) + self.broker_message
        elif single_action == 'close':
            self.close_spread()
            self.broker_message = 'new {}_CLOSE created; '.format(self.action_key) + self.broker_message

        self.last_action = single_action

