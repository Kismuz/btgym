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
from collections import deque

from btgym.strategy.utils import norm_value, decayed_result


############################## Base BTgymStrategy Class ###################


class BTgymBaseStrategy(bt.Strategy):
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

    Note:
        bt.observers.DrawDown observer will be automatically added [by server process]
        to BTgymStrategy instance at runtime.
    """
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
                low=0, # will get overridden.
                high=0,
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

    #@classmethod
    #def _set_params(cls, params):
    #    cls.params = params

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:
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
        """
        self.iteration = 1
        self.inner_embedding = 1
        self.is_done = False
        self.action = 'hold'
        self.order = None
        self.order_failed = 0
        self.broker_message = '-'
        self.raw_state = None
        self.state = dict()

        # Inherit logger from cerebro:
        self.log = self.env._log

        # Time embedding period (just take first key in obs. shapes dicts as ref.):
        self.dim_time = self.p.state_shape[list(self.p.state_shape.keys())[0]].shape[0]

        # Number of timesteps reward estimation statistics are averaged over, should be:
        # skip_frame_period < avg_period <= time_embedding_period
        self.avg_period = self.dim_time

        # Normalisation constant for statistics derived from account value:
        self.broker_value_normalizer = 1 / \
            self.env.broker.startingcash / (self.p.drawdown_call + self.p.target_call) * 100

        self.target_value = self.env.broker.startingcash * (1 + self.p.target_call / 100)

        self.trade_just_closed = False
        self.trade_result = 0

        # TODO: brush:
        self.unrealized_pnl = None
        self.norm_broker_value = None
        self.realized_pnl = None

        self.current_pos_duration = 0
        self.current_pos_min_value = 0
        self.current_pos_max_value = 0

        self.realized_broker_value = self.env.broker.startingcash
        self.episode_result = 0  # not used
        self.reward = 0

        # Service sma to get correct first features values:
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=(self.dim_time)
        )
        self.data.dim_sma.plotinfo.plot = False

        # Sliding staistics accumulators, store normalized at first place last `avg_perod` values,
        # so it's a bit more efficient than use of bt.Observers:
        sliding_datalines = [
            'broker_cash',
            'broker_value',
            'exposure',
            'leverage',
            'pos_duration',
            'realized_pnl',
            'unrealized_pnl',
            'max_unrealized_pnl',
            'min_unrealized_pnl',
        ]
        self.sliding_stat = {key: deque(maxlen=self.avg_period) for key in sliding_datalines}

        # Add custom data Lines if any (just a convenience wrapper):
        self.set_datalines()
        self.log.debug('Kwargs:\n{}\n'.format(str(kwargs)))

    def prenext(self):
        self.update_sliding_stat()

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

    def notify_trade(self, trade):
        if trade.isclosed:
            # Set trade flags: True if trade have been closed just now and within last frame-skip period,
            # and store trade result:
            self.trade_just_closed = True
            # Note: `trade_just_closed` flag has to be reset manually after evaluating.
            self.trade_result = trade.pnlcomm

            # Store realized prtfolio value:
            self.realized_broker_value = self.broker.get_value()

    def update_sliding_stat(self):
        """
        Updates all sliding statistics deques with latest-step values:
            - normalized broker value
            - normalized broker cash
            - normalized exposure (position size)
            - position duration normalized wrt. max possible episode duration
            - normalized decayed realized profit/loss for last closed trade
                (or zero if no trades been closed within last step);
            - normalized profit/loss for current opened trade (unrealized p/l);
            - normalized best possible up to present point unrealized result for current opened trade;
            - normalized worst possible up to present point unrealized result for current opened trade;
        """
        stat = self.sliding_stat
        current_value = self.env.broker.get_value()

        stat['broker_value'].append(
            norm_value(
                current_value,
                self.env.broker.startingcash,
                self.p.drawdown_call,
                self.p.target_call,
            )
        )
        stat['broker_cash'].append(
            norm_value(
                self.env.broker.get_cash(),
                self.env.broker.startingcash,
                99.0,
                self.p.target_call,
            )
        )
        stat['exposure'].append(
            self.position.size / (self.env.broker.startingcash * self.env.broker.get_leverage() + 1e-2)
        )
        stat['leverage'].append(self.env.broker.get_leverage())  # TODO: Do we need this?

        if self.trade_just_closed:
            stat['realized_pnl'].append(
                decayed_result(
                    self.trade_result,
                    current_value,
                    self.env.broker.startingcash,
                    self.p.drawdown_call,
                    self.p.target_call,
                    gamma=1
                )
            )
            # Reset flag:
            self.trade_just_closed = False
            # print('POS_OBS: step {}, just closed.'.format(self.iteration))

        else:
            stat['realized_pnl'].append(0.0)

        if self.position.size == 0:
            self.current_pos_duration = 0
            self.current_pos_min_value = current_value
            self.current_pos_max_value = current_value
            # print('ZERO_POSITION\n')

        else:
            self.current_pos_duration += 1
            if self.current_pos_max_value < current_value:
                self.current_pos_max_value = current_value

            elif self.current_pos_min_value > current_value:
                self.current_pos_min_value = current_value

        stat['pos_duration'].append(
            self.current_pos_duration / (self.data.numrecords - self.inner_embedding)
        )
        stat['max_unrealized_pnl'].append(
            (self.current_pos_max_value - self.realized_broker_value) * self.broker_value_normalizer
        )
        stat['min_unrealized_pnl'].append(
            (self.current_pos_min_value - self.realized_broker_value) * self.broker_value_normalizer
        )
        stat['unrealized_pnl'].append(
            (current_value - self.realized_broker_value) * self.broker_value_normalizer
        )

        # TODO: norm. episode duration?

        # print(
        #    'UNREALIZED: MAX_PNL:{}, MIN_PNL:{}, CURRENT_PNL: {}'.
        #        format(
        #          self.lines.max_unrealized_pnl[0],
        #          self.lines.min_unrealized_pnl[0],
        #          self.lines.unrealized_pnl[0]
        #        )
        #     )

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

        Used by renderer, at least.
        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.dim_time)),
                np.frombuffer(self.data.high.get(size=self.dim_time)),
                np.frombuffer(self.data.low.get(size=self.dim_time)),
                np.frombuffer(self.data.close.get(size=self.dim_time)),
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
        self.update_sliding_stat()
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
