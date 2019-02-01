import numpy as np
from backtrader import Indicator


class SpreadConstructor(Indicator):
    """
    Normalised Synthetic spread estimation and plotting.
    Uses norm_stat_tracker_2.
    """
    lines = ('spread',)
    plotinfo = dict(
        subplot=True,
        plotabove=True,
        plotname='Norm. Synthetic Spread',
    )
    plotlines = dict(
        spread=dict(_name='SPREAD', color='darkmagenta'),
    )

    def next(self):
        s = self._owner.norm_stat_tracker_2.get_state()
        if s.mean is None or s.variance is None:
            self.lines.spread[0] = np.random.normal(0, 0.39)

        else:
            self.lines.spread[0] = (self._owner.datas[1] - s.mean[1]) / s.variance[1] ** .5 \
                                   - (self._owner.datas[0] - s.mean[0]) / s.variance[0] ** .5


class CumSumReward(Indicator):
    """
    Cumulative reward tracking.
    """
    lines = ('cum_reward',)
    plotinfo = dict(
        subplot=True,
        plotabove=True,
        plotname='Cumulative Reward',
    )
    plotlines = dict(
        cum_reward=dict(_name='REWARD', color='darkblue'),
    )
    total_reward = 0.0

    def next(self):
        self.total_reward += self._owner.reward / self._owner.p.skip_frame
        self.lines.cum_reward[0] = self.total_reward


class SpreadSizer:
    """
    For a given pair of [supposedly co-integrated] assets
    and initial account conditions
    estimates order sizes to form a [locally] balanced "synthetic spread" position,
    (when orders are executed in opposite directions).
    Note, that it is supposed that fractional orders are supported (ok for backtrader broker)
    """

    def __init__(self, init_cash, position_max_depth, leverage=1, margin_reserve=.01):
        """

        Args:
            init_cash:              uint, initial [backtrader] broker cash
            position_max_depth:     uint, maximum compound number of same direction 'synthetic orders' to exhaust
                                    all available account cash (including unrealized broker value) w.r.t.
                                    current asset prices and co-integration coefficient.
            leverage:               float, broker leverage
            margin_reserve:         float < 1, amount to reserve in percent / 100
        """
        self.init_cash = init_cash
        self.position_max_depth = position_max_depth
        self.leverage = leverage
        self.margin_reserve = margin_reserve
        self.unit_order_size = 1
        self.alpha = 1
        # TODO: if account value rises/ drops -
        # TODO: either we scale order sizes to keep position_depth constant or  <-- currently this option
        # TODO: adjust depth to keep order sizes constant?

    def get_init_sizing(self, init_price, init_variance):
        """
        Returns position sizes w.r.t initial account value.

        Args:
            init_price:         array of floats of size [2] - initial assets prices
            init_variance:      array of floats of size [2] - initial prices variances

        Returns:
            (float, float) - position sizes for both assets
        """
        # Infer volatility-balanced order costs via current co-integrating relation,
        # which is [approximately] estimated here as ratio of standard deviations of asset prices:
        base_order1_cost = init_price[0] * init_variance[1] ** .5
        base_order2_cost = init_price[1] * init_variance[0] ** .5

        # Scaling coefficient to match required position depth:
        self.alpha = self.leverage * self.init_cash \
            / (self.unit_order_size * self.position_max_depth * (base_order1_cost + base_order2_cost))
        self.alpha *= 1 - self.margin_reserve

        base_order1_size = self.unit_order_size * init_variance[1] ** .5
        base_order2_size = self.unit_order_size * init_variance[0] ** .5

        return self.alpha * base_order1_size, self.alpha * base_order2_size

    def get_sizing(self, current_broker_value, price, variance):
        """
        Returns position sizes w.r.t current account value.

        Args:
            current_broker_value:   current account value (cash + unrealised PnL)
            price:                  array of floats of size [2] - current assets prices
            variance:               array of floats of size [2] - current prices variances

        Returns:
            (float, float) - position sizes for both assets
        """
        base_order1_cost = price[0] * variance[1] ** .5
        base_order2_cost = price[1] * variance[0] ** .5

        # Same as before but w.r.t. current value:
        self.alpha = self.leverage * current_broker_value\
            / (self.unit_order_size * self.position_max_depth * (base_order1_cost + base_order2_cost))
        self.alpha *= 1 - self.margin_reserve

        base_order1_size = self.unit_order_size * variance[1] ** .5
        base_order2_size = self.unit_order_size * variance[0] ** .5

        return self.alpha * base_order1_size, self.alpha * base_order2_size

