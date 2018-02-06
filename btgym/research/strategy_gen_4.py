import numpy as np

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
                'action': spaces.Box(low=0, high=1, shape=(avg_period, 1, 1), dtype=np.float32),
                'reward': spaces.Box(low=-1, high=1, shape=(avg_period, 1, 1), dtype=np.float32),
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
                        )
                    }
                )
            },
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
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

        self.log.debug('DEV_state_shape: {}'.format(self.p.state_shape))
        self.log.debug('DEV_skip_frame: {}'.format(self.p.skip_frame))
        self.log.debug('DEV_portfolio_actions: {}'.format(self.p.portfolio_actions))
        self.log.debug('DEV_drawdown_call: {}'.format(self.p.drawdown_call))
        self.log.debug('DEV_target_call: {}'.format(self.p.target_call))
        self.log.debug('DEV_dataset_stat:\n{}'.format(self.p.dataset_stat))
        self.log.debug('DEV_episode_stat:\n{}'.format(self.p.episode_stat))

        # Define data channels:
        self.channel_O = bt.Sum(self.data.open, - self.data.open(-1))
        self.channel_H = bt.Sum(self.data.high, - self.data.open)
        self.channel_L = bt.Sum(self.data.low,  - self.data.open)

        # Episodic metadata:
        self.state['metadata'] = {
            'type': np.asarray(self.p.metadata['type']),
            'trial_num': np.asarray(self.p.metadata['parent_sample_num']),
            'sample_num': np.asarray(self.p.metadata['sample_num']),
            'first_row': np.asarray(self.p.metadata['first_row'])
        }

    def get_market_state(self):

        x = np.stack(
            [
                np.frombuffer(self.channel_O.get(size=self.time_dim)),
                np.frombuffer(self.channel_H.get(size=self.time_dim)),
                np.frombuffer(self.channel_L.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Log-scale: NOT used. Seems to hurt performance.
        # x = log_transform(x)

        # Amplify and squash in [-1,1], seems to be best option as of 4.10.17:
        # `self.p.state_ext_scale` param is supposed to keep most of the signal
        # in 'linear' part of tanh while squashing spikes.
        x_market = tanh(x * self.p.state_ext_scale)

        return x_market[:, None, :]

    def get_broker_state(self):
        x_broker = np.concatenate(
            [
                np.asarray(self.sliding_stat['unrealized_pnl'])[..., None],
                # max_unrealized_pnl[..., None],
                # min_unrealized_pnl[..., None],
                np.asarray(self.sliding_stat['realized_pnl'])[..., None],
                np.asarray(self.sliding_stat['broker_value'])[..., None],
                np.asarray(self.sliding_stat['broker_cash'])[..., None],
                np.asarray(self.sliding_stat['exposure'])[..., None],
                # norm_episode_duration, gamma=5)[...,None],
                # norm_position_duration, gamma=2)[...,None],
            ],
            axis=-1
        )
        return x_broker[:, None, :]

    def get_state(self):

        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state['external'] = self.get_market_state()
        self.state['internal'] = self.get_broker_state()
        self.state['action'] = np.asarray(self.sliding_stat['action'])[:, None, None]
        self.state['reward'] = np.asarray(self.sliding_stat['reward'])[:, None, None]

        return self.state

    def get_reward(self):
        """
        Shapes reward function as normalized single trade realized profit/loss,
        augmented with potential-based reward shaping functions in form of:
        F(s, a, s`) = gamma * FI(s`) - FI(s);

        - potential FI_1 is current normalized unrealized profit/loss;
        - potential FI_2 is current normalized broker value.

        Paper:
            "Policy invariance under reward transformations:
             Theory and application to reward shaping" by A. Ng et al., 1999;
             http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
        """

        # All sliding statistics for this step are already updated by get_state().
        #
        # TODO: window size for stats averaging? Now it is time_dim - 1, can better be other?
        # TODO: pass actual gamma as strategy param.

        # Potential-based shaping function 1:
        # based on potential of averaged profit/loss for current opened trade (unrealized p/l):
        unrealised_pnl = np.asarray(self.sliding_stat['unrealized_pnl'])
        f1 = 1.0 * np.average(unrealised_pnl[1:]) - np.average(unrealised_pnl[:-1])

        # Potential-based shaping function 2:
        # based on potential of averaged broker value, normalized wrt to max drawdown and target bounds.
        norm_broker_value = np.asarray(self.sliding_stat['broker_value'])
        f2 = 1.0 * np.average(norm_broker_value[1:]) - np.average(norm_broker_value[:-1])

        # Main reward function: normalized realized profit/loss:
        realized_pnl = np.asarray(self.sliding_stat['realized_pnl'])[-1]

        # Weights are subject to tune:
        self.reward = 1.0 * f1 + 1.0 * f2 + 10.0 * realized_pnl
        # TODO: ------ignore-----:
        # 'Close-at-the-end' shaping term:
        # - 1.0 * self.exp_scale(avg_norm_episode_duration, gamma=6) * abs_max_norm_exposure
        # 'Do-not-expose-for-too-long' shaping term:
        # - 1.0 * self.exp_scale(avg_norm_position_duration, gamma=3)

        self.reward = np.clip(self.reward, -1, 1)

        return self.reward


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
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={}
    )

    def __init__(self, **kwargs):
        super(DevStrat_4_7, self).__init__(**kwargs)

    def get_broker_state(self):
        x_broker = np.stack(
            [
                self.sliding_stat['broker_value'][-1],
                self.sliding_stat['unrealized_pnl'][-1],
                self.sliding_stat['realized_pnl'][-1],
                self.sliding_stat['broker_cash'][-1],
                self.sliding_stat['exposure'][-1],
            ]
        )
        return x_broker[None, None, :]

    def get_state(self):
        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state['external'] = self.get_market_state()
        self.state['internal'] = self.get_broker_state()

        return self.state

    def get_reward(self):
        """
        Shapes reward function as normalized single trade realized profit/loss,
        augmented with potential-based reward shaping functions in form of:
        F(s, a, s`) = gamma * FI(s`) - FI(s);

        - potential FI_1 is current normalized unrealized profit/loss;
        - potential FI_2 is current normalized broker value.
        - FI_3: penalizing exposure toward the end of episode

        Paper:
            "Policy invariance under reward transformations:
             Theory and application to reward shaping" by A. Ng et al., 1999;
             http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
        """

        # All sliding statistics for this step are already updated by get_state().
        debug = {}

        # Potential-based shaping function 1:
        # based on potential of averaged profit/loss for current opened trade (unrealized p/l):
        unrealised_pnl = np.asarray(self.sliding_stat['unrealized_pnl'])
        f1 = self.p.gamma * np.average(unrealised_pnl[1:]) - np.average(unrealised_pnl[:-1])
        #f1 = self.p.gamma * discounted_average(unrealised_pnl[1:], self.p.gamma)\
        #     - discounted_average(unrealised_pnl[:-1], self.p.gamma)

        debug['f1'] = f1

        # Potential-based shaping function 2:
        # based on potential of averaged broker value, normalized wrt to max drawdown and target bounds.
        norm_broker_value = np.asarray(self.sliding_stat['broker_value'])
        f2 = self.p.gamma * np.average(norm_broker_value[1:]) - np.average(norm_broker_value[:-1])
        #f2 = self.p.gamma * discounted_average(norm_broker_value[1:], self.p.gamma)\
        #     - discounted_average(norm_broker_value[:-1], self.p.gamma)

        debug['f2'] = f2

        # Potential-based shaping function 3:
        # negative potential of abs. size of position, exponentially weighted wrt. episode steps
        abs_exposure = np.abs(np.asarray(self.sliding_stat['exposure']))
        time = np.asarray(self.sliding_stat['episode_step'])
        #time_w = exp_scale(np.average(time[:-1]), gamma=5)
        #time_w_prime = exp_scale(np.average(time[1:]), gamma=5)
        #f3 = - 1.0 * time_w_prime * np.average(abs_exposure[1:]) #+ time_w * np.average(abs_exposure[:-1])
        f3 = - self.p.gamma * exp_scale(time[-1], gamma=3) * abs_exposure[-1] + \
             exp_scale(time[-2], gamma=3) * abs_exposure[-2]
        debug['f3'] = f3

        # Main reward function: normalized realized profit/loss:
        realized_pnl = np.asarray(self.sliding_stat['realized_pnl'])[-1]
        debug['f_real_pnl'] = 10 * realized_pnl

        # Weights are subject to tune:
        self.reward = (1.0 * f1 + 2.0 * f2 + 0.0 * f3 + 10.0 * realized_pnl) * self.p.reward_scale

        debug['r'] = self.reward
        debug['b_v'] = self.sliding_stat['broker_value'][-1]
        debug['unreal_pnl'] = self.sliding_stat['unrealized_pnl'][-1]
        debug['iteration'] = self.iteration

        #for k, v in debug.items():
        #    print('{}: {}'.format(k, v))
        #print('\n')

        # TODO: ------ignore-----:
        # 'Do-not-expose-for-too-long' shaping term:
        # - 1.0 * self.exp_scale(avg_norm_position_duration, gamma=3)

        self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)

        return self.reward


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
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,  # not used
        metadata={},
    )

    def get_broker_state(self):
        x_broker = np.concatenate(
            [
                np.asarray(self.sliding_stat['broker_value'])[..., None],
                np.asarray(self.sliding_stat['unrealized_pnl'])[..., None],
                np.asarray(self.sliding_stat['realized_pnl'])[..., None],
                np.asarray(self.sliding_stat['broker_cash'])[..., None],
                np.asarray(self.sliding_stat['exposure'])[..., None],
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

    def get_state(self):
        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state['external'] = self.get_market_state()
        self.state['internal'] = self.get_broker_state()

        return self.state


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

    def get_market_state(self):

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

    def get_state(self):
        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state['external'] = self.get_market_state()
        self.state['internal'] = self.get_broker_state()

        return self.state


class DevStrat_4_10(DevStrat_4_7):
    """
    4_7 + Reward search: log-normalised potential functions
    """

    def get_reward(self):
        """
        Shapes reward function as normalized single trade realized profit/loss,
        augmented with potential-based reward shaping functions in form of:
        F(s, a, s`) = gamma * FI(s`) - FI(s);

        - potential FI_1 is current normalized unrealized profit/loss;
        - potential FI_2 is current normalized broker value.
        - FI_3: penalizing exposure toward the end of episode

        Paper:
            "Policy invariance under reward transformations:
             Theory and application to reward shaping" by A. Ng et al., 1999;
             http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
        """

        # All sliding statistics for this step are already updated by get_state().
        debug = {}
        scale = 10.0
        # Potential-based shaping function 1:
        # based on log potential of averaged profit/loss for current opened trade (unrealized p/l):
        unrealised_pnl = np.asarray(self.sliding_stat['unrealized_pnl']) / 2 + 1 # shift [-1,1] -> [0,1]
        # TODO: make normalizing util func to return in [0,1] by default
        f1 = self.p.gamma * np.log(np.average(unrealised_pnl[1:])) - np.log(np.average(unrealised_pnl[:-1]))

        debug['f1'] = f1

        # Potential-based shaping function 2:
        # based on potential of averaged broker value, log-normalized wrt to max drawdown and target bounds.
        norm_broker_value = np.asarray(self.sliding_stat['broker_value']) / 2 + 1 # shift [-1,1] -> [0,1]
        f2 = self.p.gamma * np.log(np.average(norm_broker_value[1:])) - np.log(np.average(norm_broker_value[:-1]))

        debug['f2'] = f2

        # Potential-based shaping function 3: NOT USED
        # negative potential of abs. size of position, exponentially weighted wrt. episode steps
        abs_exposure = np.abs(np.asarray(self.sliding_stat['exposure']))
        time = np.asarray(self.sliding_stat['episode_step'])
        #time_w = exp_scale(np.average(time[:-1]), gamma=5)
        #time_w_prime = exp_scale(np.average(time[1:]), gamma=5)
        #f3 = - 1.0 * time_w_prime * np.average(abs_exposure[1:]) #+ time_w * np.average(abs_exposure[:-1])
        f3 = - self.p.gamma * exp_scale(time[-1], gamma=3) * abs_exposure[-1] + \
             exp_scale(time[-2], gamma=3) * abs_exposure[-2]
        debug['f3'] = f3

        # `Spike` reward function: normalized realized profit/loss:
        realized_pnl = self.sliding_stat['realized_pnl'][-1]
        debug['f_real_pnl'] = 10 * realized_pnl

        # Weights are subject to tune:
        self.reward = (1.0 * f1 + 2.0 * f2 + 0.0 * f3 + 10.0 * realized_pnl) * self.p.reward_scale

        debug['r'] = self.reward
        debug['b_v'] = self.sliding_stat['broker_value'][-1]
        debug['unreal_pnl'] = self.sliding_stat['unrealized_pnl'][-1]
        debug['iteration'] = self.iteration

        #for k, v in debug.items():
        #    print('{}: {}'.format(k, v))
        #print('\n')

        # ------ignore-----:
        # 'Do-not-expose-for-too-long' shaping term:
        # - 1.0 * self.exp_scale(avg_norm_position_duration, gamma=3)

        self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)

        return self.reward


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
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        gamma=gamma,
        reward_scale=1.0,
        state_ext_scale=2e3,  # EURUSD
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

    def get_market_state(self):

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

    def get_broker_state(self):

        x_broker = np.concatenate(
            [
                np.asarray(self.sliding_stat['broker_value'])[..., None],
                np.asarray(self.sliding_stat['unrealized_pnl'])[..., None],
                np.asarray(self.sliding_stat['realized_pnl'])[..., None],
                np.asarray(self.sliding_stat['broker_cash'])[..., None],
                np.asarray(self.sliding_stat['exposure'])[..., None],
            ],
            axis=-1
        )
        x_broker = tanh(np.gradient(x_broker, axis=-1) * self.p.state_int_scale)
        return x_broker[:, None, :]


class DevStrat_4_12(DevStrat_4_11):
    """
    4_11 + sma-features 8, 512;
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

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 6), dtype=np.float32),
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
        drawdown_call=5,
        target_call=19,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        state_ext_scale=2e3,  # EURUSD
        state_int_scale=1.0,
        gamma=gamma,
        reward_scale=1.0,
        metadata={},
    )

    def set_datalines(self):
        self.data.sma_8 = btind.SimpleMovingAverage(self.datas[0], period=8)
        self.data.sma_16 = btind.SimpleMovingAverage(self.datas[0], period=16)
        self.data.sma_32 = btind.SimpleMovingAverage(self.datas[0], period=32)
        self.data.sma_64 = btind.SimpleMovingAverage(self.datas[0], period=64)
        self.data.sma_128 = btind.SimpleMovingAverage(self.datas[0], period=128)
        self.data.sma_256 = btind.SimpleMovingAverage(self.datas[0], period=256)
        #self.data.sma_512 = btind.SimpleMovingAverage(self.datas[0], period=512)

        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(512 + self.time_dim)
        )
        self.data.dim_sma.plotinfo.plot = False

    def get_market_state(self):

        x_sma = np.stack(
            [
                np.frombuffer(self.data.sma_8.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_16.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_32.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_64.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_128.get(size=self.time_dim)),
                np.frombuffer(self.data.sma_256.get(size=self.time_dim)),
                #np.frombuffer(self.data.sma_512.get(size=self.time_dim)),
            ],
            axis=-1
        )
        # Gradient along features axis:
        dx = np.gradient(x_sma, axis=-1) * self.p.state_ext_scale
        # In [-1,1]:
        x = tanh(dx)
        return x[:, None, :]

    def get_broker_state(self):

        x_broker = np.concatenate(
            [
                np.asarray(self.sliding_stat['broker_value'])[..., None],
                np.asarray(self.sliding_stat['unrealized_pnl'])[..., None],
                np.asarray(self.sliding_stat['realized_pnl'])[..., None],
                np.asarray(self.sliding_stat['broker_cash'])[..., None],
                np.asarray(self.sliding_stat['exposure'])[..., None],
            ],
            axis=-1
        )
        x_broker = tanh(np.gradient(x_broker, axis=-1) *self.p.state_int_scale)

        return x_broker[:, None, :]

