import numpy as np

import backtrader as bt
from btgym.strategy.base import BTgymBaseStrategy
from btgym.strategy.utils import tanh, abs_norm_ratio

from gym import spaces

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

        reward shaping search:
           potential-based shaping functions


    Data:
        synthetic/real
    """
    time_dim = 30  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params
    params = dict(
        # Note fake `Width` dimension to use 2d conv etc.:
        state_shape=
            {
                'external': spaces.Box(low=-1, high=1, shape=(time_dim, 1, 3)),
                'internal': spaces.Box(low=-2, high=2, shape=(time_dim, 1, 5)),
                'metadata': spaces.Dict(
                    {
                        'trial_num': spaces.Box(
                            shape=(),
                            low=0,
                            high=10**10
                        ),
                        'sample_num': spaces.Box(
                            shape=(),
                            low=0,
                            high=10**10
                        ),
                        'first_row': spaces.Box(
                            shape=(),
                            low=0,
                            high=10**10
                        )
                    }
                )
            },
        drawdown_call=5,
        target_call=19,
        portfolio_actions=('hold', 'buy', 'sell', 'close'),
        skip_frame=10,
        metadata={}
    )

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:   see BTgymBaseStrategy args.
        """
        #super(DevStrat001, self)._set_params(self.params)

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
            'trial_num': np.asarray(self.p.metadata['trial_num']),
            'sample_num': np.asarray(self.p.metadata['sample_num']),
            'first_row': np.asarray(self.p.metadata['first_row'])
        }

    def get_state(self):

        T = 2e3  # EURUSD
        # T = 1e2 # EURUSD, Z-norm
        # T = 1 # BTCUSD

        x = np.stack(
            [
                np.frombuffer(self.channel_O.get(size=self.dim_time)),
                np.frombuffer(self.channel_H.get(size=self.dim_time)),
                np.frombuffer(self.channel_L.get(size=self.dim_time)),
            ],
            axis=-1
        )
        # Log-scale: NOT used. Seems to hurt performance.
        # x = log_transform(x)

        # Amplify and squash in [-1,1], seems to be best option as of 4.10.17:
        # T param is supposed to keep most of the signal in 'linear' part of tanh while squashing spikes.
        x_market = tanh(x * T)

        # Update inner state statistic and compose state:
        self.update_sliding_stat()

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

        self.state['external'] = x_market[:, None, :]
        self.state['internal'] = x_broker[:, None, :]

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
        # TODO: pass actual gamma as strategy param. OR:  maybe: compute reward on algo side?

        # Potential-based shaping function 1:
        # based on potential of averaged profit/loss for current opened trade (unrealized p/l):
        unrealised_pnl = np.asarray(self.sliding_stat['unrealized_pnl'])
        f1 = .99 * np.average(unrealised_pnl[1:]) - np.average(unrealised_pnl[:-1])

        # Potential-based shaping function 2:
        # based on potential of averaged broker value, normalized wrt to max drawdown and target bounds.
        norm_broker_value = np.asarray(self.sliding_stat['broker_value'])
        f2 = .99 * np.average(norm_broker_value[1:]) - np.average(norm_broker_value[:-1])

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
