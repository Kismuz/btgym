import backtrader as bt


class Reward(bt.observer.Observer):
    """
    Keeps track of reward values.
    """
    lines = ('reward',)
    plotinfo = dict(plot=True, subplot=True, plotname='Reward')
    plotlines = dict(reward=dict(markersize=4.0, color='darkviolet', fillstyle='full'))

    def next(self):
        self.lines.reward[0] = self._owner.reward


class Position(bt.observer.Observer):
    """
    Keeps track of position size.
    """
    lines = ('abs_sum_exposure',)
    plotinfo = dict(plot=True, subplot=True, plotname='Position')
    plotlines = dict(abs_sum_exposure=dict(marker='.', markersize=1.0, color='blue', fillstyle='full'))

    def next(self):
        # self.lines.exposure[0] = self._owner.position.size
        # for d in self._owner.datas:
        #     print(d._name, self._owner.getposition(data=d._name))
        self.lines.abs_sum_exposure[0] = sum([abs(pos.size) for pos in self._owner.positions.values()])


# class MultiPosition(bt.observer.Observer):
#     """
#     Keeps track of position size.
#     """
#     lines = ()
#     plotinfo = dict(plot=True, subplot=True, plotname='Position')
#     plotlines = {}
#
#     def __init__(self, **kwargs):
#         self.lines = ['{}_exposure'.format(name) for name in self._owner.getdatanames()]
#         self.plotlines = {
#             line: dict(marker='.', markersize=1.0, color='blue', fillstyle='full') for line in self.lines
#         }
#         super().__init__(**kwargs)
#         print(dir(self.lines))
#
#     def next(self):
#         self.lines.exposure[0] = self._owner.position.size


class NormPnL(bt.observer.Observer):
    """
    Keeps track of PnL stats.
    """
    lines = ('realized_pnl', 'unrealized_pnl', 'max_unrealized_pnl', 'min_unrealized_pnl')
    plotinfo = dict(plot=True, subplot=True, plotname='Normalized PnL', plotymargin=.05)
    plotlines = dict(
        realized_pnl=dict(marker='o', markersize=4.0, color='blue', fillstyle='full'),
        unrealized_pnl=dict(marker='.', markersize=1.0, color='grey', fillstyle='full'),
        max_unrealized_pnl=dict(marker='.', markersize=1.0, color='c', fillstyle='full'),
        min_unrealized_pnl=dict(marker='.', markersize=1.0, color='m', fillstyle='full'),
    )

    def next(self):
        try:
            if self._owner.broker_stat['realized_pnl'][-1] != 0:
                self.lines.realized_pnl[0] = self._owner.broker_stat['realized_pnl'][-1]
        except IndexError:
            self.lines.realized_pnl[0] = 0.0

        try:
            self.lines.unrealized_pnl[0] = self._owner.broker_stat['unrealized_pnl'][-1]
        except IndexError:
            self.lines.unrealized_pnl[0] = 0.0

        try:
            self.lines.max_unrealized_pnl[0] = self._owner.broker_stat['max_unrealized_pnl'][-1]
            self.lines.min_unrealized_pnl[0] = self._owner.broker_stat['min_unrealized_pnl'][-1]

        except (IndexError, KeyError):
            self.lines.max_unrealized_pnl[0] = 0.0
            self.lines.min_unrealized_pnl[0] = 0.0
