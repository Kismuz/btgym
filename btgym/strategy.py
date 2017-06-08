###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import backtrader as bt
import backtrader.indicators as btind

import numpy as np

############################## Base BTgymStrategy Class ###################


class BTgymStrategy(bt.Strategy):
    """
    Controls Environment inner dynamics and backtesting logic.
    Any State, Reward and Info computation logic can be implemented by
    subclassing BTgymStrategy and overriding at least get_state(), get_reward(), get_info(),
    set_datalines() methods.
    One can always go deeper and override __init__ () and next() methods for desired
    server cerebro engine behaviour, including order execution etc.
    Since it is bt.Strategy subclass, see:
    https://www.backtrader.com/docu/strategy.html
    for more information.
    """

    # Set-list:
    log = None
    state = None
    reward = None
    info = '_'
    is_done = False
    iteration = 0
    action = 'hold'
    order = None
    broker_message = '-'
    params = dict(state_dim_time=10,  # state time embedding dimension (just convention)
                  state_dim_0=4,  # one can add dim_1, dim_2, ... if needed; should match env.observation_space
                  drawdown_call=20,)  # simplest condition to exit

    def __init__(self):
        # Inherit logger from cerebro:
        self.log = self.env._log

        # A wacky way to define strategy 'minimum period'
        # for proper time-embedded state composition:
        self.data.dim_sma = btind.SimpleMovingAverage(self.datas[0],
                                                      period=self.p.state_dim_time)
        # Add custom data Lines if any (just a convenience wrapper):
        self.set_datalines()

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close.
        Any other custom data lines, indicators, etc.
        should be explicitly defined by overriding this method.
        Evoked once by Strategy.__init__().
        """
        pass

    def get_state(self):
        """
        Default state observation composer.
        Returns time-embedded environment state observation as [n,m] numpy matrix, where
        n - number of signal features,
        m - time-embedding length.
        One can override this method,
        defining necessary calculations and return arbitrary shaped tensor.
        It's possible either to compute entire featurized environment state
        or just pass raw price data to RL algorithm featurizer module.
        Note1: 'data' referes to bt.startegy datafeeds and should be treated as such.
        Datafeed Lines that are not default to BTgymStrategy should be explicitly defined in
        define_datalines().
        Note2: 'n' is essentially == env.state_dim_0.
        """
        self.state = np.row_stack((self.data.open.get(size=self.p.state_dim_time),
                                   self.data.low.get(size=self.p.state_dim_time),
                                   self.data.high.get(size=self.p.state_dim_time),
                                   self.data.close.get(size=self.p.state_dim_time),))

    def get_reward(self):
        """
        Default reward estimator.
        Same as for state composer applies. Can return raw portfolio
        performance statictics or enclose entire reward estimation algorithm.
        """
        self.reward = (self.stats.broker.value[0] - self.stats.broker.value[-1]) * 1e2

    def get_info(self):
        """
        Composes information part of environment response,
        can be any string/object. Override by own taste.
        """
        self.info = ('Step: {}\nAgent action: {}\n' +
                     'Portfolio Value: {:.5f}\n' +
                     'Reward: {:.4f}\n' +
                     '{}\n' +  # broker message is here
                     'Drawdown: {:.4f}\n' +
                     'Max.Drawdown: {:.4f}\n').format(self.iteration,
                                                      self.action,
                                                      self.stats.broker.value[0],
                                                      self.reward,
                                                      self.broker_message,
                                                      self.stats.drawdown.drawdown[0],
                                                      self.stats.drawdown.maxdrawdown[0])

    def get_done(self):
        """
        Default episode termination estimator, checks conditions episode stop is called upon,
        <self.is_done> flag is also used as part of environment response.
        """
        # Prepare for the worst and run checks:
        self.is_done = True
        # Will it be last step of the episode?:
        if self.iteration >= self.data.numrecords - self.p.state_dim_time:
            self.broker_message = 'END OF DATA!'
        elif self.action == '_done':
            self.broker_message = '_DONE SIGNAL RECEIVED'
        # Any money left?:
        elif self.stats.drawdown.maxdrawdown[0] > self.p.drawdown_call:
            self.broker_message = 'DRAWDOWN CALL!'
        # ...............
        # Finally, it seems ok to continue:
        else:
            self.is_done = False
            return
        # Or else, initiate fallback to Control Mode; still executes strategy cycle once:
        self.log.debug('RUNSTOP() evoked with {}'.format(self.broker_message))
        self.env.runstop()

    def notify_order(self, order):
        """
        Shamelessly taken from backtrader tutorial.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.broker_message = 'BUY executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.broker_message = 'SELL executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled]:
            self.broker_message = 'Order Canceled'
        elif order.status in [order.Margin, order.Rejected]:
            self.broker_message = 'Order Margin/Rejected'
        self.order = None

    def next(self):
        """
        Default implementation.
        Defines one step environment routine for server 'Episode mode';
        At least, it should handle order execution logic according to action received and
        evoke following methods:
                self.get_done()
                self.get_state()
                self.get_reward()
                self.get_info()  - this should be on last place.
        """
        # Housekeeping:
        self.iteration += 1
        self.broker_message = '-'

        # How we're doing?:
        self.get_done()

        if not self.is_done:
            # All the next() computations should be performed here [as function of <self.action>],
            # this ensures async. server/client execution.
            # Note: that implies that action execution is lagged for 1 step.
            # <is_done> flag can also be rised here by trading logic events,
            # e.g. <OMG! We became too rich!>/<Hide! Black Swan is coming!>

            # Simple action-to-order logic:
            if self.action == 'hold' or self.order:
                pass
            elif self.action == 'buy':
                self.order = self.buy()
                self.broker_message = 'New BUY created; ' + self.broker_message
            elif self.action == 'sell':
                self.order = self.sell()
                self.broker_message = 'New SELL created; ' + self.broker_message
            elif self.action == 'close':
                self.order = self.close()
                self.broker_message = 'New CLOSE created; ' + self.broker_message
        else:  # time to leave:
            self.close()

        # Gather response:
        self.get_state()
        self.get_reward()
        self.get_info()

        # Somewhere at this point, server-side _EpisodeComm() is exchanging information with environment wrapper,
        # obtaining <self.action> and sending <state,reward,done,info>... never mind.
