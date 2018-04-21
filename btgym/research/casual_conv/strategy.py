import numpy as np

from gym import spaces
from btgym import DictSpace
import backtrader.indicators as btind

from btgym.strategy.utils import tanh
from btgym.research.gps.strategy import GuidedStrategy_0_0


class CasualConvStrategy(GuidedStrategy_0_0):
    """
    Provides stream of data for casual convolutional encoder
    """
    # Time embedding period:
    time_dim = 128  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Hyperparameters for estimating signal features:
    features_parameters = [8, 32, 128]
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

    state_ext_scale = np.linspace(3e3, 1e3, num=num_features)

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, num_features), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'expert': spaces.Box(low=0, high=1, shape=(len(portfolio_actions),), dtype=np.float32),  # TODO: change inheritance!
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
                    )
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

