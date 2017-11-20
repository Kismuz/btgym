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
    RnD grade code.

    Objectives:
        external state data feature search:
            time_embedded three-channeled vector:
                - `Open` channel is one time-step difference of Open price;
                - `High` and `Low` channels are differences
                  between current Open price and current High or Low prices respectively

        internal state data feature search:
            time_embedded concatenated vector of broker and portfolio statistics

        reward shaping search:
            weighted sum of averaged over time_embedding period broker and portfolio statisitics.


    Data:
        synthetic/real

    Note:

        call classmethod set_params() before use!
    """
    time_dim = 30
    params = dict(
        state_shape=
            {
                'external': spaces.Box(low=-1, high=1, shape=(time_dim, 3)),
                'internal': spaces.Box(low=-2, high=2, shape=(time_dim, 5)),
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
        self.state['external'] = tanh(x * T)

        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state['internal'] = np.concatenate(
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
        #self.state['raw_state'] = self.raw_state
        return self.state

    def get_reward(self):

        # All reward terms for this step are already updated by get_state().

        # Reward term 1: averaged profit/loss for current opened trade (unrealized p/l):
        avg_unrealised_pnl = np.average(self.sliding_stat['unrealized_pnl'])

        # Reward term 2: averaged broker value, normalized wrt to max drawdown and target bounds.
        avg_norm_broker_value = np.average(self.sliding_stat['broker_value'])

        # Reward term 3: normalized realized profit/loss:

        # Check if any trades been closed in given period:
        realized_pnl = np.asarray(self.sliding_stat['realized_pnl'])
        is_result = realized_pnl!= 0

        # If yes - compute averaged result:
        if is_result.any():
            avg_realized_pnl = np.average(realized_pnl[is_result])
            # Realised-to-possible-result weight,
            # e.g rate at scale [0.1, 2] how achieved result relates to best/worst possible trade scenario:
            max_to_real_k = np.clip(
                abs_norm_ratio(
                    avg_realized_pnl,
                    np.average(self.sliding_stat['min_unrealized_pnl']),
                    np.average(self.sliding_stat['max_unrealized_pnl']),
                ),
                0.1,
                2
            )
            # print('max_to_real_k:{}, x:{}\na:{}\nb:{}'.
            # format(
            #    max_to_real_k,
            #    avg_realized_pnl,
            #    np.average(self.min_unrealized_pnl),
            #    np.average(self.max_unrealized_pnl),
            #    )
            # )
        else:
            avg_realized_pnl = 0
            max_to_real_k = 1

        # avg_norm_episode_duration = ...
        # abs_max_norm_exposure = ...
        # avg_norm_position_duration = ...

        # Weights are subject to tune:
        self.reward = (
            + 1.0 * avg_unrealised_pnl * max_to_real_k  # TODO: make it zero when pos.size=0
            + 0.01 * avg_norm_broker_value
            + 10.0 * avg_realized_pnl * max_to_real_k
            # 'Close-at-the-end' term:
            # - 1.0 * self.exp_scale(avg_norm_episode_duration, gamma=6) * abs_max_norm_exposure
            # 'Do-not-expose-for-too-long' term:
            # - 1.0 * self.exp_scale(avg_norm_position_duration, gamma=3)
        )
        self.reward = np.clip(self.reward, -1, 1)

        return self.reward