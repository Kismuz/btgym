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
from btgym import DictSpace

import numpy as np
from collections import deque

from btgym.strategy.utils import norm_value, decayed_result, exp_scale


############################## Base BTgymStrategy Class ###################


class BTgymBaseStrategy(bt.Strategy):
    """
    Controls Environment inner dynamics and backtesting logic. Provides gym'my (State, Action, Reward, Done, Info) data.
    Any State, Reward and Info computation logic can be implemented by subclassing BTgymStrategy and overriding
    get_[mode]_state(), get_reward(), get_info(), is_done() and set_datalines() methods.
    One can always go deeper and override __init__ () and next() methods for desired
    server cerebro engine behaviour, including order execution logic etc.

    Note:
        - base class supports single asset iteration via default data_line named 'base_asset', see derived classes
          multi-asset support
        - bt.observers.DrawDown observer will be automatically added to BTgymStrategy instance at runtime.
        - Since it is bt.Strategy subclass, refer to https://www.backtrader.com/docu/strategy.html for more information.
    """

    # Time embedding period:
    time_dim = 4  # NOTE: changed this --> change Policy  UNREAL for aux. pix control task upsampling params

    # Number of environment steps to skip before returning next response,
    # e.g. if set to 10 -- agent will interact with environment every 10th step;
    # every other step agent action is assumed to be 'hold':
    skip_frame = 1

    # Number of timesteps reward estimation statistics are averaged over, should be:
    # skip_frame_period <= avg_period <= time_embedding_period:
    avg_period = time_dim

    # Possible agent actions;  Note: place 'hold' first! :
    portfolio_actions = ('hold', 'buy', 'sell', 'close')

    params = dict(
        # Observation state shape is dictionary of Gym spaces,
        # at least should contain `raw_state` field.
        # By convention first dimension of every Gym Box space is time embedding one;
        # one can define any shape; should match env.observation_space.shape.
        # observation space state min/max values,
        # For `raw_state' (default) - absolute min/max values from BTgymDataset will be used.
        state_shape={
            'raw': spaces.Box(
                shape=(time_dim, 4),
                low=0, # will get overridden.
                high=0,
                dtype=np.float32,
            ),
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
        cash_name='default_cash',
        asset_names=['default_asset'],
        start_cash=None,
        commission=None,
        leverage=1.0,
        drawdown_call=10,  # finish episode when hitting drawdown treshghold , in percent.
        target_call=10,  # finish episode when reaching profit target, in percent.
        dataset_stat=None,  # Summary descriptive statistics for entire dataset and
        episode_stat=None,  # current episode. Got updated by server.
        metadata={},
        trial_stat=None,
        trial_metadata=None,
        portfolio_actions=portfolio_actions,
        skip_frame=skip_frame,
        order_size=None,
        initial_action=None,
        initial_portfolio_action=None,
    )

    def __init__(self, **kwargs):
        """
        Keyword Args:
            params (dict):          parameters dictionary, see Note below.


            Notes:
                Due to backtrader convention, any strategy arguments should be defined inside `params` dictionary
                or passed as kwargs to bt.Cerebro() class via .addstrategy() method. Parameter dictionary
                should contain at least these keys::

                    state_shape:        Observation state shape is dictionary of Gym spaces, by convention
                                        first dimension of every Gym Box space is time embedding one;
                    cash_name:          str, name for cash asset
                    asset_names:        iterable of str, names for assets
                    start_cash:         float, broker starting cash
                    commission:         float, broker commission value, .01 stands for 1%
                    leverage:           broker leverage, default is 1.0
                    order_size:         dict of fixed order stakes (floats); keys should match assets names.
                    drawdown_call:      finish episode when hitting this drawdown treshghold , in percent.
                    target_call:        finish episode when reaching this profit target, in percent.
                    portfolio_actions:  possible agent actions.
                    skip_frame:         number of environment steps to skip before returning next response,
                                        e.g. if set to 10 -- agent will interact with environment every 10th step;
                                        every other step agent action is assumed to be 'hold'.

                Default values are::

                    state_shape=dict(raw_state=spaces.Box(shape=(4, 4), low=0, high=0,))
                    cash_name='default_cash'
                    asset_names=['default_asset']
                    start_cash=None
                    commission=None
                    leverage=1.0
                    drawdown_call=10
                    target_call=10
                    dataset_stat=None
                    episode_stat=None
                    portfolio_actions=('hold', 'buy', 'sell', 'close')
                    skip_frame=1
                    order_size=None
        """
        try:
            self.time_dim = self.p.state_shape['raw'].shape[0]
        except KeyError:
            pass

        try:
            self.skip_frame = self.p.skip_frame
        except KeyError:
            pass
        
        self.iteration = 0
        self.env_iteration = 0
        self.inner_embedding = 1
        self.is_done = False
        self.is_done_enabled = False
        self.steps_till_is_done = 2  # extra steps to make when episode terminal conditions are met
        self.action = self.p.initial_portfolio_action
        self.last_action = self.p.initial_portfolio_action
        self.reward = 0
        self.order = None
        self.order_failed = 0
        self.broker_message = '_'
        self.final_message = '_'
        self.raw_state = None
        self.time_stamp = 0

        # Inherit logger from cerebro:
        self.log = self.env._log

        # Prepare broker:
        if self.p.start_cash is not None:
            self.env.broker.setcash(self.p.start_cash)

        if self.p.commission is not None:
            self.env.broker.setcommission(commission=self.p.commission, leverage=self.p.leverage)

        # Normalisation constant for statistics derived from account value:
        self.broker_value_normalizer = 1 / \
            self.env.broker.startingcash / (self.p.drawdown_call + self.p.target_call) * 100

        self.target_value = self.env.broker.startingcash * (1 + self.p.target_call / 100)

        # Try to define stake, if no self.p.order_size dict has been set:
        if self.p.order_size is None:
            # If no order size has been set for every data_line,
            # try to infer stake size from sizer set by bt.Cerebro.addsizer() method:
            try:
                assert len(list(self.env.sizers.values())) == 1
                env_sizer_params = list(self.env.sizers.values())[0][-1]  # pull dict of outer set sizer params
                assert 'stake' in env_sizer_params.keys()

            except (AssertionError, KeyError) as e:
                msg = 'Order stake is not set neither via strategy.param.order_size nor via bt.Cerebro.addsizer method.'
                self.log.error(msg)
                raise ValueError(msg)

            self.p.order_size = {name: env_sizer_params['stake'] for name in self.p.asset_names}

        elif isinstance(self.p.order_size, int) or isinstance(self.p.order_size, float):
            unimodal_stake = {name: self.p.order_size for name in self.p.asset_names}
            self.p.order_size = unimodal_stake

        self.trade_just_closed = False
        self.trade_result = 0

        self.unrealized_pnl = None
        self.norm_broker_value = None
        self.realized_pnl = None

        self.current_pos_duration = 0
        self.current_pos_min_value = 0
        self.current_pos_max_value = 0

        self.realized_broker_value = self.env.broker.startingcash
        self.episode_result = 0  # not used

        # Service sma to get correct first features values:
        self.data.dim_sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=self.time_dim
        )
        self.data.dim_sma.plotinfo.plot = False

        # self.log.warning('self.p.dir: {}'.format(dir(self.params)))

        # Episode-wide metadata:
        self.metadata = {
            'type': np.asarray(self.p.metadata['type']),
            'trial_num': np.asarray(self.p.metadata['parent_sample_num']),
            'trial_type': np.asarray(self.p.metadata['parent_sample_type']),
            'sample_num': np.asarray(self.p.metadata['sample_num']),
            'first_row': np.asarray(self.p.metadata['first_row']),
            'timestamp': np.asarray(self.time_stamp, dtype=np.float64)
        }
        self.state = {
            'raw': None,
            'metadata': None
        }

        # This flag shows to the outer world if this episode can move global
        # time forward (see: btgym.server._BTgymAnalyzer.next() method);
        # default logic: iff it is test episode from target domain:
        self.can_increment_global_time = self.metadata['type'] and self.metadata['trial_type']

        self.log.debug('strategy.metadata: {}'.format(self.metadata))
        self.log.debug('can_increment_global_time: {}'.format(self.can_increment_global_time))

        # Sliding staistics accumulators, globally normalized last `avg_perod` values,
        # so it's a bit more comp. efficient than use of bt.Observers:
        sliding_datalines = [
            'broker_cash',
            'broker_value',
            'exposure',
            'leverage',
            'pos_duration',
            'episode_step',
            'realized_pnl',
            'unrealized_pnl',
            'max_unrealized_pnl',
            'min_unrealized_pnl',
            # 'action',
            'reward',
        ]
        self.sliding_stat = {key: deque(maxlen=self.avg_period) for key in sliding_datalines}

        # Add custom data Lines if any (convenience wrapper):
        self.set_datalines()
        self.log.debug('Kwargs:\n{}\n'.format(str(kwargs)))

        # Here we define collection dictionary looking for methods for estimating state, one method for one mode,
        # should be named .get_[mode_name]_state():
        self.collection_get_state_methods = {}
        for key in self.p.state_shape.keys():
            try:
                self.collection_get_state_methods[key] = getattr(self, 'get_{}_state'.format(key))

            except AttributeError:
                raise NotImplementedError('Callable get_{}_state.() not found'.format(key))

        for data in self.datas:
            self.log.debug('data_name: {}'.format(data._name))

        self.log.debug('stake size: {}'.format(self.p.order_size))

        # Define how this strategy should handle actions: either as discrete or continuous:
        if self.p.portfolio_actions is None or set(self.p.portfolio_actions) == {}:
            # No discrete actions provided, assume continuous:
            self.next_process_fn = self._next_target_percent
            # Disable broker checking margin,
            # see: https://community.backtrader.com/topic/152/multi-asset-ranking-and-rebalancing/2?page=1
            self.env.broker.set_checksubmit(False)

        else:
            # Use discrete handling method otherwise:
            self.next_process_fn = self._next_discrete

    def prenext(self):
        self.update_sliding_stat()

    def nextstart(self):
        self.inner_embedding = self.data.close.buflen()
        self.log.debug('Inner time embedding: {}'.format(self.inner_embedding))

    def next(self):
        """
        Default implementation for built-in backtrader method.
        Defines one step environment routine;
        At least, it should handle order execution logic according to action received.
        Note that orders can only be submitted for data_lines in action_space (assets).
        `self.action` attr. is updated by btgym.server._BTgymAnalyzer, and `None` actions
        are emitted while doing `skip_frame` loop.
        """
        if '_skip_this' in self.action.keys():
                # print('a_skip, b_message: ', self.broker_message)
                return

        else:
            self.next_process_fn(self.action)
            # print('a_process, b_message: ', self.broker_message)

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
            - position duration in steps, normalized wrt. max possible episode steps
            - exp. scaled episode duration in steps, normalized wrt. max possible episode steps
            - normalized realized profit/loss for last closed trade (is zero if no pos. closures within last env. step)
            - normalized profit/loss for current opened trade (unrealized p/l);
            - normalized best possible up to present point unrealized result for current opened trade;
            - normalized worst possible up to present point unrealized result for current opened trade;
            - DELETED: one hot encoding for actions received;
            - rewards received (based on self.reward variable values);
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
        stat['episode_step'].append(
            exp_scale(
                self.iteration / (self.data.numrecords - self.inner_embedding),
                gamma=3
            )
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
        #stat['action'].append(self.action_norm(self.last_action))
        stat['reward'].append(self.reward)

        #print(stat['episode_step'])

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close, Volume.
        Any other custom data lines, indicators, etc. should be explicitly defined by overriding this method.
        Invoked once by Strategy.__init__().
        """
        pass

    def get_raw_state(self):
        """
        Default state observation composer.

        Returns:
             and updates time-embedded environment state observation as [n,4] numpy matrix, where:
                4 - number of signal features  == state_shape[1],
                n - time-embedding length  == state_shape[0] == <set by user>.

        Note:
            `self.raw_state` is used to render environment `human` mode and should not be modified.

        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.high.get(size=self.time_dim)),
                np.frombuffer(self.data.low.get(size=self.time_dim)),
                np.frombuffer(self.data.close.get(size=self.time_dim)),
            )
        ).T

        return self.raw_state

    def get_metadata_state(self):
        self.metadata['timestamp'] = np.asarray(self._get_timestamp())

        return self.metadata

    def _get_time(self):
        """
        Retrieves current time point of the episode data.

        Returns:
            datetime object
        """
        return self.data.datetime.datetime()

    def _get_timestamp(self):
        """
        Sets attr. and returns current data timestamp.

        Returns:
            POSIX timestamp
        """
        self.time_stamp = self._get_time().timestamp()

        return self.time_stamp

    def get_state(self):
        """
        Collects estimated values for every mode of observation space by calling methods from
        `collection_get_state_methods` dictionary.
        As a rule, this method should not be modified, override or implement correspondig get_[mode]_state() methods,
        defining necessary calculations and return arbitrary shaped tensors for every space mode.

        Note:
            - 'data' referes to bt.startegy datafeeds and should be treated as such.
                Datafeed Lines that are not default to BTgymStrategy should be explicitly defined by
                 __init__() or define_datalines().
        """
        # Update inner state statistic and compose state:
        self.update_sliding_stat()

        self.state = {key: method() for key, method in self.collection_get_state_methods.items()}
        # Above line is generalisation of, say:
        # self.state = {
        #     'external': self.get_external_state(),
        #     'internal': self.get_internal_state(),
        #     'datetime': self.get_datetime_state(),
        #     'metadata': self.get_metadata_state(),
        # }
        return self.state

    def get_reward(self):
        """
        Default reward estimator.

        Computes `dummy` reward as log utility of current to initial portfolio value ratio.
        Same principles as for state composer apply.

        Returns:
             self.reward value: scalar, float

        Note:
            should update self.reward variable.
        """
        self.reward = float(np.log(self.stats.broker.value[0] / self.env.broker.startingcash))
        return self.reward

    def get_info(self):
        """
        Composes information part of environment response,
        can be any object. Override to own taste.

        Note:
            Due to 'skip_frame' feature, INFO part of environment response transmitted by server can be  a list
            containing either all skipped frame's info objects, i.e. [info[-9], info[-8], ..., info[0]] or
            just latest one, [info[0]]. This behaviour is set inside btgym.server._BTgymAnalyzer().next() method.
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
        defines any trading logic conditions episode stop is called upon, e.g. <OMG! Stop it, we became too rich!>.
        It is just a structural a convention method. Default method is empty.

        Expected to return:
            tuple (<is_done, type=bool>, <message, type=str>).
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
            4. Hit target profit threshold.

        This method shouldn't be overridden or called explicitly.

        Runtime execution logic is:
            terminate episode if:
                get_done() returned (True, 'something')
                OR
                ANY _get_done() default condition is met.
        """
        if not self.is_done_enabled:
            # Episode is on its way,
            # apply base episode termination rules:
            is_done_rules = [
                # Do we approaching the end of the episode?:
                (self.iteration >= \
                 self.data.numrecords - self.inner_embedding - self.p.skip_frame - self.steps_till_is_done,
                 'END OF DATA'),
                # Any money left?:
                (self.stats.drawdown.maxdrawdown[0] >= self.p.drawdown_call, 'DRAWDOWN CALL'),
                # Party time?
                (self.env.broker.get_value() > self.target_value, 'TARGET REACHED'),
            ]
            # Append custom get_done() results, if any:
            is_done_rules += [self.get_done()]

            # Sweep through rules:
            for (condition, message) in is_done_rules:
                if condition:
                    # Start episode termination countdown for clean exit:
                    # to forcefully execute final `close` order and compute proper reward
                    # we need to make `steps_till_is_done` number of steps until `is_done` flag can be safely risen:
                    self.is_done_enabled = True
                    self.broker_message += message
                    self.final_message = message
                    self.order = self.close()
                    self.log.debug(
                        'Episode countdown started at: {}, {}, r:{}'.format(self.iteration, message, self.reward)
                    )

        else:
            # Now in episode termination phase,
            # just keep hitting `Close` button:
            self.steps_till_is_done -=1
            self.broker_message = 'CLOSE, {}'.format(self.final_message)
            self.order = self.close()
            self.log.debug(
                'Episode countdown contd. at: {}, {}, r:{}'.format(self.iteration, self.broker_message, self.reward)
            )

        if self.steps_till_is_done <= 0:
            # Now we've done, terminate:
            self.is_done = True

        return self.is_done

    def notify_order(self, order):
        """
        Shamelessly taken from backtrader tutorial.
        TODO: better multi data support
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

    def _next_discrete(self, action):
        """
        Default implementation for discrete actions.
        Note that orders can be submitted only for data_lines in action_space (assets).

        Args:
            action:     dict, string encoding of btgym.spaces.ActionDictSpace

        """
        for key, single_action in action.items():
            # Simple action-to-order logic:
            if single_action == 'hold' or self.is_done_enabled:
                pass
            elif single_action == 'buy':
                self.order = self.buy(data=key, size=self.p.order_size[key])
                self.broker_message = 'new {}_BUY created; '.format(key) + self.broker_message
            elif single_action == 'sell':
                self.order = self.sell(data=key, size=self.p.order_size[key])
                self.broker_message = 'new {}_SELL created; '.format(key) + self.broker_message
            elif single_action == 'close':
                self.order = self.close(data=key)
                self.broker_message = 'new {}_CLOSE created; '.format(key) + self.broker_message

        # Somewhere after this point, server-side _BTgymAnalyzer() is exchanging information with environment wrapper,
        # obtaining <self.action> , composing and sending <state,reward,done,info> etc... never mind.

    def _next_target_percent(self, action):
        """
        Uses `order_target_percent` method to rebalance assets to given ratios. Expects action for every asset to be
        a float scalar in [0,1], with actions sum to 1 over all assets (including base one).
        Note that action for base asset (cash) is ignored.
        For details refer to: https://www.backtrader.com/docu/order_target/order_target.html
        """
        # TODO 1: filter similar actions to prevent excessive orders issue e.g by DKL on two consecutive ones
        # TODO 2: actions discretesation on level of execution
        for asset in self.p.asset_names:
                # Reducing assets positions subj to 5% margin reserve:
                single_action = round(float(action[asset]) * 0.9, 2)
                self.order = self.order_target_percent(data=asset, target=single_action )
                self.broker_message += ' new {}->{:1.0f}% created; '.format(asset, single_action * 100)

