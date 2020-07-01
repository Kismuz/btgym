import numpy as np

import backtrader as bt
from btgym.research.strategy_gen_4 import DevStrat_4_12
from btgym.research.gps.oracle import Oracle, Oracle2

from gym import spaces
from btgym import DictSpace


class GuidedStrategy_0_0(DevStrat_4_12):
    """
    Augments observation state with expert actions predictions estimated by accessing entire episode data (=cheating).
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

    state_ext_scale = np.linspace(3e3, 1e3, num=6)

    params = dict(
        # Note: fake `Width` dimension to use 2d conv etc.:
        state_shape=
        {
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 6), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 5), dtype=np.float32),
            'datetime': spaces.Box(low=0, high=1, shape=(1, 5), dtype=np.float32),
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
        # Expert parameters:
        expert_config=
        {
            'time_threshold': 5,  # minimum peak estimation radius in number of environment steps
            'pips_threshold': 5,  # minimum peak estimation value in number of quota points
            'pips_scale': 1e-4,   # value of single quota point relative to price value
            'kernel_size': 5,     # gaussian_over_action tails size in number of env. steps
            'kernel_stddev': 1,   # gaussian_over_action standard deviation
        },
    )

    def __init__(self, **kwargs):
        super(GuidedStrategy_0_0, self).__init__(**kwargs)
        self.expert = Oracle(action_space=np.arange(len(self.p.portfolio_actions)), **self.p.expert_config)
        # self.expert = Oracle2(action_space=np.arange(len(self.p.portfolio_actions)), **self.p.expert_config)
        self.expert_actions = None
        self.current_expert_action = None

    def nextstart(self):
        """
        Overrides base method augmenting it with estimating expert actions before actual episode starts.
        """
        # This value shows how much episode records we need to spend
        # to estimate first environment observation:
        self.inner_embedding = self.data.close.buflen()
        self.log.info('Inner time embedding: {}'.format(self.inner_embedding))

        # Now when we know exact maximum possible episode length -
        #  can extract relevant episode data and make expert predictions:
        data = self.datas[0].p.dataname.values[self.inner_embedding:, :]

        # Note: need to form sort of environment 'custom candels' by taking min and max price values over every
        # skip_frame period; this is done inside Oracle class;
        # TODO: shift actions forward to eliminate one-point prediction lag?
        # expert_actions is a matrix representing discrete distribution over actions probabilities
        # of size [max_env_steps, action_space_size]:
        self.expert_actions = self.expert.fit(episode_data=data, resampling_factor=self.p.skip_frame)

    def get_expert_state(self):
        self.current_expert_action = self.expert_actions[self.env_iteration]

        #print('Strat_iteration:', self.iteration)
        #print('Env_iteration:', self.env_iteration)

        return self.current_expert_action

    # def get_state(self):
    #     # Update inner state statistic and compose state:
    #     self.update_broker_stat()
    #
    #     self.state = {
    #         'external': self.get_external_state(),
    #         'internal': self.get_internal_state(),
    #         'datetime': self.get_datetime_state(),
    #         'expert': self.get_expert_state(),
    #         'metadata': self.get_metadata_state(),
    #     }
    #
    #     return self.state


class ExpertObserver(bt.observer.Observer):
    """
    Keeps track of expert-advised actions.
    Single data_feed.
    """

    lines = ('buy', 'sell', 'hold', 'close')
    plotinfo = dict(plot=True, subplot=True, plotname='Expert Actions', plotymargin=.8)
    plotlines = dict(
        buy=dict(marker='^', markersize=4.0, color='cyan', fillstyle='full'),
        sell=dict(marker='v', markersize=4.0, color='magenta', fillstyle='full'),
        hold=dict(marker='.', markersize=1.0, color='gray', fillstyle='full'),
        close=dict(marker='o', markersize=4.0, color='blue', fillstyle='full')
    )

    def next(self):
        action = np.argmax(self._owner.current_expert_action)
        if action == 0:
            self.lines.hold[0] = 0
        elif action == 1:
            self.lines.buy[0] = 1
        elif action == 2:
            self.lines.sell[0] = -1
        elif action == 3:
            self.lines.close[0] = 0

