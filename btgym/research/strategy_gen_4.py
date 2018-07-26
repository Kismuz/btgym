import numpy as np
from scipy.stats import zscore

import backtrader as bt
import backtrader.indicators as btind

from btgym.strategy.base import BTgymBaseStrategy
from btgym.strategy.utils import tanh, abs_norm_ratio, exp_scale, discounted_average, log_transform

from gym import spaces
from btgym import DictSpace

"""
Research grade code. Can be unstable, buggy, poor performing and generally is subject to change.
"""


class DevStrat_4_6(BTgymBaseStrategy):
    """
    Objectives:
        external state data feature search:
            time_embedded three-channeled vector:
                - `Open` channel is one time-step difference of Open price;
                - `High` and `Low` channels are differences
                  between current Open price and current High or Low prices respectively

        internal state data feature search:
            time_embedded concatenated vector of broker and portfolio statistics
            time_embedded vector of last actions recieved (one-hot)
            time_embedded vector of rewards

        reward shaping search:
           potential-based shaping functions

    Data:
        synthetic/real
    """

    # Time embedding period:
    time_dim = 30  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = time_dim

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
            {
                'external': spaces.Box(low=-1, high=1, shape=(time_dim, 1, 3), dtype=np.float32),
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
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={}
    )

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:   see BTgymBaseStrategy args.
        """
        super(DevStrat_4_6, self).__init__(**kwargs)
        self.state['metadata'] = self.metadata

        self.log.debug('DEV_state_shape: {}'.format(self.p.state_shape))
        self.log.debug('DEV_skip_frame: {}'.format(self.p.skip_frame))
        self.log.debug('DEV_portfolio_actions: {}'.format(self.p.portfolio_actions))
        self.log.debug('DEV_drawdown_call: {}'.format(self.p.drawdown_call))
        self.log.debug('DEV_target_call: {}'.format(self.p.target_call))
        self.log.debug('DEV_dataset_stat:\n{}'.format(self.p.dataset_stat))
        self.log.debug('DEV_episode_stat:\n{}'.format(self.p.episode_stat))

    def set_datalines(self):

        # Define data channels:
        self.channel_O = bt.Sum(self.data.open, - self.data.open(-1))
        self.channel_H = bt.Sum(self.data.high, - self.data.open)
        self.channel_L = bt.Sum(self.data.low, - self.data.open)

    def get_external_state(self):

        x = np.stack(
            [
                np.frombuffer(self.channel_O.get(size=self.time_dim)),
                np.frombuffer(self.channel_H.get(size=self.time_dim)),
                np.frombuffer(self.channel_L.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Amplify and squash in [-1,1], seems to be best option as of 4.10.17:
        # `self.p.state_ext_scale` param is supposed to keep most of the signal
        # in 'linear' part of tanh while squashing spikes.
        x_market = tanh(x * self.p.state_ext_scale)

        return x_market[:, None, :]


class DevStrat_4_7(DevStrat_4_6):
    """
    4_6 + Sliding statistics avg_period disentangled from time embedding dim;
    Only one last step sliding stats are used for internal state;
    Reward weights: 1, 2, 10 , reward scale factor aded;
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

    gamma = 1.0  # fi_gamma, should be MDP gamma decay

    reward_scale = 1.0  # reward scaler

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-1, high=1, shape=(time_dim, 1, 3), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(1, 1, 5), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={}
    )

    def __init__(self, **kwargs):
        super(DevStrat_4_7, self).__init__(**kwargs)

    def get_internal_state(self):
        x_broker = np.stack(
            [
                self.broker_stat['value'][-1],
                self.broker_stat['unrealized_pnl'][-1],
                self.broker_stat['realized_pnl'][-1],
                self.broker_stat['cash'][-1],
                self.broker_stat['exposure'][-1],
            ]
        )
        return x_broker[None, None, :]


class DevStrat_4_8(DevStrat_4_7):
    """
    4_7 + Uses full average_period of inner stats for use with inner_conv_encoder.
    """
    # Time embedding period:
    time_dim = 30  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 10

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    # !..-> here it is also `broker state` time-embedding period
    avg_period = 20

    # Possible agent actions:
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    gamma = 1.0  # fi_gamma, should be MDP gamma decay, but somehow undiscounted works better <- wtf?

    reward_scale = 1  # reward multiplicator

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-1, high=1, shape=(time_dim, 1, 3), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={},
    )

    def get_internal_state(self):
        x_broker = np.concatenate(
            [
                np.asarray(self.broker_stat['value'])[..., None],
                np.asarray(self.broker_stat['unrealized_pnl'])[..., None],
                np.asarray(self.broker_stat['realized_pnl'])[..., None],
                np.asarray(self.broker_stat['cash'])[..., None],
                np.asarray(self.broker_stat['exposure'])[..., None],
                # np.asarray(self.sliding_stat['episode_step'])[..., None],
                # np.asarray(self.sliding_stat['reward'])[..., None],
                # np.asarray(self.sliding_stat['action'])[..., None],
                # norm_position_duration[...,None],
                # max_unrealized_pnl[..., None],
                # min_unrealized_pnl[..., None],
            ],
            axis=-1
        )
        return x_broker[:, None, :]


class DevStrat_4_9(DevStrat_4_7):
    """
    4_7 + Uses simple SMA market state features.
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

    gamma = 1.0  # fi_gamma, should be MDP gamma decay

    reward_scale = 1  # reward multiplicator, touchy!

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 8), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(1, 1, 5), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=1e4,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={},
    )

    def set_datalines(self):
        self.data.sma_4 = btind.SimpleMovingAverage(self.datas[0], period=4)
        self.data.sma_8 = btind.SimpleMovingAverage(self.datas[0], period=8)
        self.data.sma_16 = btind.SimpleMovingAverage(self.datas[0], period=16)
        self.data.sma_32 = btind.SimpleMovingAverage(self.datas[0], period=32)
        self.data.sma_64 = btind.SimpleMovingAverage(self.datas[0], period=64)
        self.data.sma_128 = btind.SimpleMovingAverage(self.datas[0], period=128)
        self.data.sma_256 = btind.SimpleMovingAverage(self.datas[0], period=256)

        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(256 + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_external_state(self):

        x = np.stack(
            [
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_4.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_8.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_16.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_32.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_64.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_128.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_256.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Gradient along features axis:
        x = np.gradient(x, axis=1) * self.p.state_ext_scale

        # Log-scale:
        x = log_transform(x)
        return x[:, None, :]


class DevStrat_4_10(DevStrat_4_7):
    """
    4_7 + Reward search: log-normalised potential functions. Nope.
    """

    # def get_reward(self):
    #     """
    #     Shapes reward function as normalized single trade realized profit/loss,
    #     augmented with potential-based reward shaping functions in form of:
    #     F(s, a, s`) = gamma * FI(s`) - FI(s);
    #
    #     - potential FI_1 is current normalized unrealized profit/loss;
    #     - potential FI_2 is current normalized broker value.
    #     - FI_3: penalizing exposure toward the end of episode
    #
    #     Paper:
    #         "Policy invariance under reward transformations:
    #          Theory and application to reward shaping" by A. Ng et al., 1999;
    #          http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
    #     """
    #
    #     # All sliding statistics for this step are already updated by get_state().
    #     debug = {}
    #     scale = 10.0
    #     # Potential-based shaping function 1:
    #     # based on log potential of averaged profit/loss for current opened trade (unrealized p/l):
    #     unrealised_pnl = np.asarray(self.broker_stat['unrealized_pnl']) / 2 + 1 # shift [-1,1] -> [0,1]
    #     # TODO: make normalizing util func to return in [0,1] by default
    #     f1 = self.p.gamma * np.log(np.average(unrealised_pnl[1:])) - np.log(np.average(unrealised_pnl[:-1]))
    #
    #     debug['f1'] = f1
    #
    #     # Potential-based shaping function 2:
    #     # based on potential of averaged broker value, log-normalized wrt to max drawdown and target bounds.
    #     norm_broker_value = np.asarray(self.broker_stat['value']) / 2 + 1 # shift [-1,1] -> [0,1]
    #     f2 = self.p.gamma * np.log(np.average(norm_broker_value[1:])) - np.log(np.average(norm_broker_value[:-1]))
    #
    #     debug['f2'] = f2
    #
    #     # Potential-based shaping function 3: NOT USED
    #     # negative potential of abs. size of position, exponentially weighted wrt. episode steps
    #     # abs_exposure = np.abs(np.asarray(self.broker_stat['exposure']))
    #     # time = np.asarray(self.broker_stat['episode_step'])
    #     # #time_w = exp_scale(np.average(time[:-1]), gamma=5)
    #     # #time_w_prime = exp_scale(np.average(time[1:]), gamma=5)
    #     # #f3 = - 1.0 * time_w_prime * np.average(abs_exposure[1:]) #+ time_w * np.average(abs_exposure[:-1])
    #     # f3 = - self.p.gamma * exp_scale(time[-1], gamma=3) * abs_exposure[-1] + \
    #     #      exp_scale(time[-2], gamma=3) * abs_exposure[-2]
    #     # debug['f3'] = f3
    #     f3 = 1
    #
    #     # `Spike` reward function: normalized realized profit/loss:
    #     realized_pnl = self.broker_stat['realized_pnl'][-1]
    #     debug['f_real_pnl'] = 10 * realized_pnl
    #
    #     # Weights are subject to tune:
    #     self.reward = (1.0 * f1 + 2.0 * f2 + 0.0 * f3 + 10.0 * realized_pnl) * self.p.reward_scale
    #
    #     debug['r'] = self.reward
    #     debug['b_v'] = self.broker_stat['value'][-1]
    #     debug['unreal_pnl'] = self.broker_stat['unrealized_pnl'][-1]
    #     debug['iteration'] = self.iteration
    #
    #     #for k, v in debug.items():
    #     #    print('{}: {}'.format(k, v))
    #     #print('\n')
    #
    #     # ------ignore-----:
    #     # 'Do-not-expose-for-too-long' shaping term:
    #     # - 1.0 * self.exp_scale(avg_norm_position_duration, gamma=3)
    #
    #     self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)
    #
    #     return self.reward


class DevStrat_4_11(DevStrat_4_10):
    """
    4_10 + Another set of sma-features, grads for broker state
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

    gamma = 0.99  # fi_gamma, should be MDP gamma decay

    reward_scale = 1  # reward multiplicator

    state_ext_scale = np.linspace(3e3, 1e3, num=5)

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 5), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 6), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        metadata={},
    )

    def set_datalines(self):
        self.data.sma_16 = btind.SimpleMovingAverage(self.datas[0], period=16)
        self.data.sma_32 = btind.SimpleMovingAverage(self.datas[0], period=32)
        self.data.sma_64 = btind.SimpleMovingAverage(self.datas[0], period=64)
        self.data.sma_128 = btind.SimpleMovingAverage(self.datas[0], period=128)
        self.data.sma_256 = btind.SimpleMovingAverage(self.datas[0], period=256)

        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(256 + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_external_state(self):

        x_sma = np.stack(
            [
                np.frombuffer(self.data.sma_16.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_32.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_64.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_128.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_256.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Gradient along features axis:
        dx = np.gradient(x_sma, axis=-1) * self.p.state_ext_scale

        x = tanh(dx)

        return x[:, None, :]

    def get_internal_state(self):

        x_broker = np.concatenate(
            [
                np.asarray(self.broker_stat['value'])[..., None],
                np.asarray(self.broker_stat['unrealized_pnl'])[..., None],
                np.asarray(self.broker_stat['realized_pnl'])[..., None],
                np.asarray(self.broker_stat['cash'])[..., None],
                np.asarray(self.broker_stat['exposure'])[..., None],
                np.asarray(self.broker_stat['pos_direction'])[..., None],

                # np.asarray(self.broker_stat['value'])[-self.p.skip_frame:, None],
                # np.asarray(self.broker_stat['unrealized_pnl'])[-self.p.skip_frame:, None],
                # np.asarray(self.broker_stat['realized_pnl'])[-self.p.skip_frame:, None],
                # np.asarray(self.broker_stat['cash'])[-self.p.skip_frame:, None],
                # np.asarray(self.broker_stat['exposure'])[-self.p.skip_frame:, None],
                # np.asarray(self.broker_stat['pos_direction'])[-self.p.skip_frame:, None],
            ],
            axis=-1
        )
        x_broker = tanh(np.gradient(x_broker, axis=-1) * self.p.state_int_scale)
        # return x_broker[:, None, :]
        return np.clip(x_broker[:, None, :], -2, 2)


class DevStrat_4_11_1(DevStrat_4_11):
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
    gamma = 0.99  # fi_gamma, should be MDP gamma decay
    reward_scale = 1  # reward multiplicator
    state_ext_scale = np.linspace(3e3, 1e3, num=5)
    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': DictSpace(
                {
                    'diff': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 5), dtype=np.float32),
                    'avg': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 5), dtype=np.float32),
                }
            ),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 6), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=state_ext_scale,  # EURUSD
        state_int_scale=1.0,
        metadata={},
    )

    def get_external_state(self):
        x_sma = np.stack(
            [
                np.frombuffer(self.data.sma_16.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_32.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_64.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_128.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_256.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Gradient along features axis:
        diff = np.gradient(x_sma, axis=-1) * self.p.state_ext_scale
        diff = tanh(diff)
        avg = np.gradient(x_sma, axis=0) * self.p.state_ext_scale
        avg = tanh(avg)

        return {'avg': avg[:, None, :], 'diff': diff[:, None, :]}

class DevStrat_4_12(DevStrat_4_11):
    """
    4_11 + sma-features 8, 512;
    """
    # Time embedding period:
    time_dim = 30  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Hyperparameters for estimating signal features:
    features_parameters = [8, 16, 32, 64, 128, 256]
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
            'datetime': spaces.Box(low=0, high=1, shape=(1, 5), dtype=np.float32),
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
        leverage=1.0,
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        initial_action=None,
        initial_portfolio_action=None,
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

    def get_external_state(self):

        x_sma = np.stack(
            [
                feature.get(size=self.time_dim) for feature in self.data.features
            ],
            axis=-1
        )
        # Gradient along features axis:
        dx = np.gradient(x_sma, axis=-1) * self.p.state_ext_scale

        # In [-1,1]:
        x = tanh(dx)
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

    def get_datetime_state(self):
        time = self.data.datetime.time()
        date = self.data.datetime.date()

        # Encode in [0, 1]:
        mn = date.month / 12
        wd = date.weekday() / 6
        d = date.day / 31
        h = time.hour / 24
        mm = time.minute / 60

        encoded_stamp = [mn, d, wd, h, mm]
        return np.asarray(encoded_stamp)[None, :]


