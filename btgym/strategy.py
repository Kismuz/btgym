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

from gym import spaces

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

    # Set-list. We don't to fiddle with subclassing bt.Cerebro(),
    # so BTgymStrategy will contain all backtrading engine parameters
    # and attributes one can need at runtime:
    log = None
    iteration = 1
    inner_embedding = 1
    is_done = False
    action = 'hold'
    order = None
    order_failed = 0
    broker_message = '-'
    raw_state = None
    state = dict()
    params = dict(
        # Observation state shape is dictionary of Gym spaces,
        # at least should contain `raw_state` field.
        # By convention first dimension of every Gym Box space is time embedding one;
        # one can define any shape; should match env.observation_space.shape.
        # observation space state min/max values,
        # For `raw_state' - absolute min/max values from BTgymDataset will be used.
        state_shape=dict(
            raw_state=spaces.Box(
                shape=(10, 4),
                low=-100, # will get overridden.
                high=100,
            )
        ),
        drawdown_call=10,  # finish episode when hitting drawdown treshghold , in percent.
        target_call=10,  # finish episode when reaching profit target, in percent.
        dataset_stat=None,  # Summary descriptive statistics for entire dataset and
        episode_stat=None,  # current episode. Got updated by server.
        portfolio_actions=('hold', 'buy', 'sell', 'close'),  # possible agent actions.
        skip_frame=1,  # Number of environment steps to skip before returning next response,
                       # e.g. if set to 10 -- agent will interact with environment every 10th episode step;
                       # Every other step agent action is assumed to be 'hold'.
    )

    def __init__(self, **kwargs):
        # Inherit logger from cerebro:
        self.log = self.env._log

        # A wacky way to define strategy 'minimum period'
        # for proper time-embedded state composition:
        self.data.dim_sma = btind.SimpleMovingAverage(self.datas[0],
                                                      period=self.p.state_shape['raw_state'].shape[0])
        self.data.dim_sma.plotinfo.plot = False
        self.target_value = self.env.broker.startingcash * (1 + self.p.target_call / 100)

        # Add custom data Lines if any (just a convenience wrapper):
        self.set_datalines()
        self.log.debug('Kwargs:\n{}\n'.format(str(kwargs)))

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close, Volume.
        Any other custom data lines, indicators, etc.
        should be explicitly defined by overriding this method [convention].
        Invoked once by Strategy.__init__().
        """
        #self.log.warning('Deprecated method. Use __init__  with Super(..., self).__init__(**kwargs) instead.')

    def _get_raw_state(self):
        """
        Default state observation composer.
        Returns time-embedded environment state observation as [n,4] numpy matrix, where
        4 - number of signal features  == state_shape[1],
        n - time-embedding length  == state_shape[0] == <set by user>.
        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.p.state_shape['raw_state'].shape[0])),
                np.frombuffer(self.data.low.get(size=self.p.state_shape['raw_state'].shape[0])),
                np.frombuffer(self.data.high.get(size=self.p.state_shape['raw_state'].shape[0])),
                np.frombuffer(self.data.close.get(size=self.p.state_shape['raw_state'].shape[0])),
            )
        ).T

        return self.raw_state

    def get_state(self):
        """
        One can override this method,
        defining necessary calculations and return arbitrary shaped tensor.
        It's possible either to compute entire featurized environment state
        or just pass raw price data to RL algorithm featurizer module.
        Note1: 'data' referes to bt.startegy datafeeds and should be treated as such.
        Datafeed Lines that are not default to BTgymStrategy should be explicitly defined in
        define_datalines().
        NOTE: while iterating, ._get_raw_state() method is called just before this one,
        so variable `self.raw_state` is fresh and ready to use.
        """
        self.state['raw_state'] = self.raw_state
        return self.state

    def get_reward(self):
        """
        Default reward estimator.
        Computes reward as log utility of current to initial portfolio value ratio.
        Returns scalar <reward, type=float>.
        Same principles as for state composer apply. Can return raw portfolio
        performance statistics or enclose entire reward estimation algorithm.
        """
        return float(np.log(self.stats.broker.value[0] / self.env.broker.startingcash))

    def get_info(self):
        """
        Composes information part of environment response,
        can be any object. Override to own taste.
        Note: Due to 'skip_frame' feature,
        INFO part of environment response will be a list of all skipped frame's info objects,
        i.e. [info[-9], info[-8], ..., info[0].
        """
        return dict(
            step=self.iteration,
            time=self.data.datetime.datetime(),
            action=self.action,
            broker_message=self.broker_message,
            broker_cash=self.stats.broker.cash[0],
            broker_value=self.stats.broker.value[0],
            drawdown=self.stats.drawdown.drawdown[0],
            max_drawdown=self.stats.drawdown.maxdrawdown[0],
        )

    def get_done(self):
        """
        Episode termination estimator,
        defines any trading logic conditions episode stop is called upon,
        e.g. <OMG! Stop it, we became too rich!> .
        It is just a structural a convention method.
        Default method is empty.
        Expected to return tuple (<is_done, type=bool>, <message, type=str>).
        """
        return False, '-'

    def _get_done(self):
        """
        Default episode termination method,
        checks base conditions episode stop is called upon:
        1. Reached maximum episode duration. Need to check it explicitly, because <self.is_done> flag
           is sent as part of environment response.
        2. Got '_done' signal from outside. E.g. via env.reset() method invoked by outer RL algorithm.
        3. Hit drawdown threshold.
        4. Hit target profit treshhold.

        This method shouldn't be overridden or called explicitly.

        Runtime execution logic is:
            terminate episode if:
            get_done() returned (True, 'something')
            OR
            ANY _get_done() default condition is met.
        """
        # Base episode termination rules:
        is_done_rules = [
            # Will it be last step of the episode?:
            (self.iteration >= self.data.numrecords - self.inner_embedding, 'END OF DATA!'),
            # Any money left?:
            (self.stats.drawdown.maxdrawdown[0] >= self.p.drawdown_call, 'DRAWDOWN CALL!'),
            # Party time?
            (self.env.broker.get_value() > self.target_value, 'TARGET REACHED!'),
        ]

        # Append custom get_done() results, if any:
        is_done_rules += [self.get_done()]

        # Sweep through rules:
        for (condition, message) in is_done_rules:
            if condition:
                self.is_done = True
                self.broker_message = message
                self.order = self.close()
        return self.is_done

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

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.broker_message = 'ORDER FAILED with status: ' + str(order.getstatusname())
            # Rise order_failed flag until get_reward() will [hopefully] use and reset it:
            self.order_failed += 1
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

        # Somewhere after this point, server-side _BTgymAnalyzer() is exchanging information with environment wrapper,
        # obtaining <self.action> , composing and sending <state,reward,done,info> etc... never mind.
