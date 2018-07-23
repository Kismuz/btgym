import numpy as np
import scipy.signal as signal
from scipy.stats import zscore

import backtrader as bt
import backtrader.indicators as btind

from btgym.strategy.base import BTgymBaseStrategy
from btgym.strategy.utils import tanh, abs_norm_ratio, exp_scale, discounted_average, log_transform

from gym import spaces
from btgym import DictSpace

from btgym.research.strategy_gen_4 import DevStrat_4_10

class DevStrat_2_0(DevStrat_4_10):
    """
    Get back to CWT:
    As 4_10, but computes observation state
    by applying continious wavelet transform to time-embedded vector
    to hi/low median price gradient.
    """
    # Time embedding period:
    time_dim = 30  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = 20

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    gamma = 0.99  # fi_gamma, should be equal MDP gamma decay

    reward_scale = 1  # reward scaling scalar

    cwt_signal_scale = 2e3  # first gradient scaling [scalar]
    cwt_lower_bound = 3.0   # CWT scales
    cwt_upper_bound = 12.0

    state_ext_scale = np.linspace(3, 1, num=5)  # second gradient scaling [vector]

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 5), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
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
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        cwt_lower_bound=cwt_lower_bound,
        cwt_upper_bound=cwt_upper_bound,
        cwt_signal_scale=cwt_signal_scale,
        metadata={},
    )

    def __init__(self, **kwargs):
        super(DevStrat_2_0, self).__init__(**kwargs)
        self.num_channels = self.p.state_shape['external'].shape[-1]
        # Define CWT scales:
        self.cwt_width = np.linspace(self.p.cwt_lower_bound, self.p.cwt_upper_bound, self.num_channels)

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

        # Note: differences taken once again along channels axis,
        # apply weighted scaling to normalize channels
        norm_x = np.gradient(cwt_x, axis=-1)
        norm_x = zscore(norm_x, axis=0) * self.p.state_ext_scale
        #out_x = tanh(norm_x)
        out_x = np.clip(norm_x, -10, 10)

        return out_x[:, None, :]

    def get_internal_state(self):
        x_broker = np.concatenate(
            [
                np.asarray(self.broker_stat['broker_value'])[..., None],
                np.asarray(self.broker_stat['unrealized_pnl'])[..., None],
                np.asarray(self.broker_stat['realized_pnl'])[..., None],
                np.asarray(self.broker_stat['broker_cash'])[..., None],
                np.asarray(self.broker_stat['exposure'])[..., None],
            ],
            axis=-1
        )
        x_broker = tanh(np.gradient(x_broker, axis=-1) * self.p.state_int_scale)
        return x_broker[:, None, :]
