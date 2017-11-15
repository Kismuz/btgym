import backtrader as bt


class Reward(bt.observer.Observer):
    """
    Keeps track of reward values.
    """
    lines = ('reward',)
    plotinfo = dict(plot=True, subplot=True)
    plotlines = dict(reward=dict(markersize=4.0, color='darkviolet', fillstyle='full'))

    def next(self):
        self.lines.reward[0] = self._owner.reward


class Position(bt.observer.Observer):
    """
    Keeps track of position size.
    """
    lines = ('exposure',)
    plotinfo = dict(plot=True, subplot=True)
    plotlines = dict(exposure=dict(marker='.', markersize=1.0, color='blue', fillstyle='full'))

    def next(self):
        self.lines.exposure[0] = self._owner.position.size


class NormPnL(bt.observer.Observer):
    """
    Keeps track of PnL stats.
    """
    lines = ('realized_pnl', 'unrealized_pnl', 'max_unrealized_pnl', 'min_unrealized_pnl')
    plotinfo = dict(plot=True, subplot=True)
    plotlines = dict(
        realized_pnl=dict(marker='.', markersize=1.0, color='blue', fillstyle='full'),
        unrealized_pnl=dict(marker='.', markersize=1.0, color='grey', fillstyle='full'),
        max_unrealized_pnl=dict(marker='.', markersize=1.0, color='c', fillstyle='full'),
        min_unrealized_pnl=dict(marker='.', markersize=1.0, color='m', fillstyle='full'),
    )

    def next(self):
        self.lines.realized_pnl[0] = self._owner.sliding_stat['realized_pnl'][-1]
        self.lines.unrealized_pnl[0] = self._owner.sliding_stat['unrealized_pnl'][-1]
        self.lines.max_unrealized_pnl[0] = self._owner.sliding_stat['max_unrealized_pnl'][-1]
        self.lines.min_unrealized_pnl[0] = self._owner.sliding_stat['min_unrealized_pnl'][-1]