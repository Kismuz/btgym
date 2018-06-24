import numpy as np

from gym import spaces
from btgym import DictSpace
import backtrader.indicators as btind
from backtrader import Indicator

from btgym.strategy.utils import tanh
from btgym.research.gps.strategy import GuidedStrategy_0_0


class CasualConvStrategy(GuidedStrategy_0_0):
    """
    Provides stream of data for casual convolutional encoder
    """
    # Time embedding period:
    time_dim = 128  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Hyperparameters for estimating signal features:
    # features_parameters = [8, 32, 64]
    features_parameters = [8, 32, 128, 512]
    num_features = len(features_parameters)

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 20

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    gamma = 0.99  # fi_gamma, should be MDP gamma decay

    reward_scale = 1  # reward multiplicator

    state_ext_scale = np.linspace(4e3, 1e3, num=num_features)

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, num_features), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'datetime': spaces.Box(low=0, high=1, shape=(1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),  # TODO: change inheritance!
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
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        gamma=gamma,
        reward_scale=1.0,
        metadata={},
    )

    def set_datalines(self):
        self.data.features = [
            btind.SimpleMovingAverage(self.datas[0], period=period) for period in self.features_parameters
        ]

        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(np.asarray(self.features_parameters).max() + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False


class MaxPool(Indicator):
    """
    Custom period `sliding candle` upper bound.
    """
    lines = ('max',)
    params = (('period', 1),)
    plotinfo = dict(
        subplot=False,
        plotlinevalues=False,
    )

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        self.lines.max[0] = np.frombuffer(self.data.high.get(size=self.p.period)).max()


class MinPool(Indicator):
    """
    Custom period `sliding candle` lower bound.
    """
    lines = ('min',)
    params = (('period', 1),)
    plotinfo = dict(
        subplot=False,
        plotlinevalues=False,
    )

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        self.lines.min[0] = np.frombuffer(self.data.low.get(size=self.p.period)).min()


class CasualConvStrategy_0(CasualConvStrategy):
    """
    Casual convolutional encoder + `sliding candle` price data features instead of SMA.
    """
    # Time embedding period:
    # time_dim = 512  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params
    time_dim = 128
    # time_dim = 32

    # Periods for estimating signal features,
    # note: here number of feature channels is doubled due to fact Hi/Low values computed for each period specified:

    # features_parameters = [8, 32, 128, 512]
    # features_parameters = [2, 8, 32, 64, 128]
    features_parameters = [8, 16, 32, 64, 128, 256]

    num_features = len(features_parameters)

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are collected over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 20

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    gamma = 0.99  # fi_gamma, should be MDP gamma decay

    reward_scale = 1  # reward multiplicator

    state_ext_scale = np.linspace(2e3, 1e3, num=num_features)

    params = dict(
        # Note: fake `Width` dimension to stay in convention with 2d conv. dims:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, num_features * 2), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'datetime': spaces.Box(low=0, high=1, shape=(1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),
        # TODO: change inheritance!
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
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        gamma=gamma,
        reward_scale=1.0,
        metadata={},
    )

    def set_datalines(self):
        features_low = [MinPool(self.data, period=period) for period in self.features_parameters]
        features_high = [MaxPool(self.data, period=period) for period in self.features_parameters]

        # If `scale` was scalar - make it vector:
        if len(np.asarray(self.p.state_ext_scale).shape) < 1:
            self.p.state_ext_scale = np.repeat(np.asarray(self.p.state_ext_scale), self.num_features)

        # Sort features by `period` for .get_external_state() to estimate
        # more or less sensible gradient; double-stretch scale vector accordingly:
        # TODO: maybe 2 separate conv. encoders for hi/low?
        self.data.features = []
        for f1, f2 in zip(features_low, features_high):
            self.data.features += [f1, f2]

        self.p.state_ext_scale = np.repeat(self.p.state_ext_scale, 2)

        # print('p.state_ext_scale: ', self.p.state_ext_scale, self.p.state_ext_scale.shape)

        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(np.asarray(self.features_parameters).max() + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False


import scipy.signal as signal
from scipy.stats import zscore


class CasualConvStrategy_1(CasualConvStrategy_0):
    """
    CWT. again.
    """
    # Time embedding period:
    # time_dim = 512
    # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params
    # NOTE_2: should be power of 2 if using casual conv. state encoder
    time_dim = 128
    # time_dim = 32

    # Periods for estimating signal features,
    # note: here number of feature channels is doubled due to fact Hi/Low values computed for each period specified:

    # features_parameters = [8, 32, 128, 512]
    # features_parameters = [2, 8, 32, 64, 128]
    # features_parameters = [8, 16, 32, 64, 128, 256]
    #
    # num_features = len(features_parameters)

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are collected over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period
    # NOTE_: should be power of 2 if using casual conv. state encoder:
    avg_period = 20

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    gamma = 0.99  # fi_gamma, should be MDP gamma decay

    reward_scale = 1  # reward multiplicator

    num_features = 16

    cwt_signal_scale = 3e3  # first gradient scaling [scalar]
    cwt_lower_bound = 3.0   # CWT scales
    cwt_upper_bound = 90.0

    state_ext_scale = np.linspace(1, 3, num=num_features)

    params = dict(
        # Note: fake `Width` dimension to stay in convention with 2d conv. dims:
        state_shape=
        {
            'raw': spaces.Box(low=-100, high=100, shape=(time_dim, 4), dtype=np.float32),
            # 'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, num_features), dtype=np.float32),
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, num_features, 1), dtype=np.float32),
            # 'external_2': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 4), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'datetime': spaces.Box(low=0, high=1, shape=(1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=10, shape=(len(portfolio_actions),), dtype=np.float32),
            # TODO: change inheritance!
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
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        gamma=gamma,
        reward_scale=1.0,
        metadata={},
        cwt_lower_bound=cwt_lower_bound,
        cwt_upper_bound=cwt_upper_bound,
        cwt_signal_scale=cwt_signal_scale,
    )

    def __init__(self, **kwargs):
        super(CasualConvStrategy_1, self).__init__(**kwargs)
        # self.num_channels = self.p.state_shape['external'].shape[-1]
        self.num_channels = self.num_features
        # Define CWT scales:
        self.cwt_width = np.linspace(self.p.cwt_lower_bound, self.p.cwt_upper_bound, self.num_channels)

    def set_datalines(self):
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(np.asarray(self.features_parameters).max() + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_external_state(self):
        # Use Hi-Low median as signal:
        x = (
            np.frombuffer(self.data.high.get(size=self.time_dim)) +
            np.frombuffer(self.data.low.get(size=self.time_dim))
        ) / 2

        # Differences along time dimension:
        d_x = np.gradient(x, axis=0) * self.p.cwt_signal_scale

        # Compute continuous wavelet transform using Ricker wavelet:
        cwt_x = signal.cwt(d_x, signal.ricker, self.cwt_width).T

        norm_x = cwt_x

        # Note: differences taken once again along channels axis,
        # apply weighted scaling to normalize channels
        # norm_x = np.gradient(cwt_x, axis=-1)
        # norm_x = zscore(norm_x, axis=0) * self.p.state_ext_scale
        # norm_x *= self.p.state_ext_scale

        out_x = tanh(norm_x)

        # out_x = np.clip(norm_x, -10, 10)

        # return out_x[:, None, :]
        return out_x[..., None]

    def get_external_2_state(self):
        x = np.stack(
            [
                np.frombuffer(self.data.high.get(size=self.time_dim)),
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.low.get(size=self.time_dim)),
                np.frombuffer(self.data.close.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # # Differences along features dimension:
        d_x = np.gradient(x, axis=-1) * self.p.cwt_signal_scale

        # Compute continuous wavelet transform using Ricker wavelet:
        # cwt_x = signal.cwt(d_x, signal.ricker, self.cwt_width).T

        norm_x = d_x

        # Note: differences taken once again along channels axis,
        # apply weighted scaling to normalize channels
        # norm_x = np.gradient(cwt_x, axis=-1)
        # norm_x = zscore(norm_x, axis=0) * self.p.state_ext_scale
        # norm_x *= self.p.state_ext_scale

        out_x = tanh(norm_x)

        # out_x = np.clip(norm_x, -10, 10)

        return out_x[:, None, :]

