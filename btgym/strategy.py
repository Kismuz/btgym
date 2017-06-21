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
    subclassing BTgymStrategy and overriding at least get_state(), get_reward(),
    get_info(), is_done() and set_datalines() methods.
    One can always go deeper and override __init__ () and next() methods for desired
    server cerebro engine behaviour, including order execution etc.
    Since it is bt.Strategy subclass, see:
    https://www.backtrader.com/docu/strategy.html
    for more information.
    Note: bt.observers.DrawDown observer will be automatically added [by server process]
    to BTgymStrategy instance at runtime.
    """

    # Set-list:
    log = None
    state = None
    reward = None
    info = '_'
    is_done = False
    iteration = 1
    action = 'hold'
    order = None
    broker_message = '-'
    params = dict(state_dim_time=10,  # state time embedding dimension (just convention)
                  state_dim_0=4,  # one can add dim_1, dim_2, ... if needed; should match env.observation_space
                  state_low=None,  # observation space state min/max values,
                  state_high=None,  # if set to None - absolute min/max values from BTgymDataset will be used.
                  drawdown_call=90,)  # simplest condition to finish episode

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
        should be explicitly defined by overriding this method [convention].
        Invoked once by Strategy.__init__().
        """
        pass

    def get_state(self):
        """
        Default state observation composer.
        Returns time-embedded environment state observation as [n,m] numpy matrix, where
        n - number of signal features [ == env.state_dim_0],
        m - time-embedding length.
        One can override this method,
        defining necessary calculations and return arbitrary shaped tensor.
        It's possible either to compute entire featurized environment state
        or just pass raw price data to RL algorithm featurizer module.
        Note1: 'data' referes to bt.startegy datafeeds and should be treated as such.
        Datafeed Lines that are not default to BTgymStrategy should be explicitly defined in
        define_datalines().
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
        # Let it be 1-step portfolio value delta:
        # TODO: make it more sensible
        self.reward = (self.stats.broker.value[0] - self.stats.broker.value[-1]) * 1e2

    def get_info(self):
        """
        Composes information part of environment response,
        can be any object. Override by own taste.
        """
        self.info = dict(step = self.iteration,
                         action = self.action,
                         broker_message = self.broker_message,
                         broker_value = self.stats.broker.value[0],
                         drawdown = self.stats.drawdown.drawdown[0],
                         max_drawdown = self.stats.drawdown.maxdrawdown[0],)

    def get_done(self):
        """
        Episode termination estimator,
        defines any trading logic conditions episode stop is called upon,
        e.g. <OMG! Stop it, we became too rich!> .
        It is just a structural a convention method.
        If any desired condition is met, it should set <self.is_done> variable to True,
        and [optionaly] set <self.broker_message> to some info string.
        Episode runtime termination logic is:
        ANY <get_done condition is met> OR ANY <_get_done() default condition is met>
        """
        pass

    def _get_done(self):
        """
        Default episode termination method,
        checks base conditions episode stop is called upon:
        1. Reached maximum episode duration. Need to check it explicitly, because <self.is_done> flag
           is sent as part of environment response.
        2. Got '_done' signal from outside. E.g. via env.reset() method invoked by outer RL algorithm.
        3. Hit drawdown threshold.
        """

        # Base episode termination rules:
        is_done_rules = [
            # Will it be last step of the episode?:
            (self.iteration >= self.data.numrecords - self.p.state_dim_time, 'END OF DATA!'),
            # If agent/server forces episode termination?:
            (self.action == '_done', '_DONE SIGNAL RECEIVED'),
            # Any money left?:
            (self.stats.drawdown.maxdrawdown[0] > self.p.drawdown_call, 'DRAWDOWN CALL!'),
        ]

        # Sweep through:
        for (condition, message) in is_done_rules:
            if condition:
                self.is_done = True
                self.broker_message = message


    def notify_order(self, order):
        """
        Shamelessly taken from backtrader tutorial.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
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
        At least, it should handle order execution logic according to action received.
        """

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

        # Somewhere after this point, server-side _EpisodeComm() is exchanging information with environment wrapper,
        # obtaining <self.action> , composing and sending <state,reward,done,info> etc... never mind.
