###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin, muzikinae@gmail.com
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

import logging
#logging.basicConfig(format='%(name)s: %(message)s')
import multiprocessing
import time
import datetime
import random
import itertools
import zmq
import importlib

import os

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding, closer

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind


############################## Environment part ##############################

class BacktraderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_filename=None, # TODO: check file actually exists
                 datafeed_params=None,
                 port=5500,
                 min_episode_len=2000,
                 max_episode_days=2,
                 custom_strategy_class=None,
                 state_embed_dim=10,
                 state_dim_1=4,
                 # TODO: drawdown_call param, etc.
                 portfolio_actions=['hold', 'buy', 'sell', 'close'],
                 verbose=False, ):

        # Check datafeed existence:
        if not os.path.isfile(data_filename):
            raise FileNotFoundError('Datafeed not found: ' + data_filename)

        # Verbosity control:
        self.log = logging.getLogger('Env')
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        self.verbose = verbose

        # Server/network parameters:
        self.server = None
        self.port = port
        self.network_address = 'tcp://127.0.0.1:{}'.format(port)

        # Set client channel:
        # first, kill any process using server port:
        # cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.port)
        # os.system(cmd)
        # ZMQ!:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.network_address)

        # Episode data and computattion logic related parameters:
        self.min_episode_len = min_episode_len
        self.max_episode_days = max_episode_days
        self.custom_strategy_class = custom_strategy_class

        # Observation space:
        self.state_embed_dim = state_embed_dim
        self.state_dim_1 = state_dim_1
        self.observation_space = spaces.Box(low=0.0,
                                            high=10.0,
                                            shape=(self.state_dim_1,
                                                   self.state_embed_dim))

        # Action space and corresponding server messages:
        self.action_space = spaces.Discrete(len(portfolio_actions))
        self.server_actions = portfolio_actions + ['_done', '_reset', '_stop']
        """
        Default parsing parameters for source-specific CSV datafeed class.
        Correctly parses 1 minute Forex generic ASCII
        data files from www.HistData.com:
        """
        if not datafeed_params:
            self.datafeed_params = dict(
                dataname=data_filename,
                # nullvalue= 0.0,
                fromdate=None,
                todate=None,
                dtformat='%Y%m%d %H%M%S',
                headers=False,
                separator=';',
                timeframe=1,
                datetime=0,
                high=1,
                low=2,
                open=3,
                close=4,
                volume=5,
                openinterest=-1,
                numrecords=0,  # just keep it here.
                info='1 min FX generic ASCII, www.HistData.com', )
        else:
            self.datafeed_params = datafeed_params
            self.datafeed_params['dataname'] = data_filename

        self.log.info('Environment is ready.')

    def _start_server(self):
        """
        Configures backtrader REQ/REP server instance and starts server process.
        """
        self.server = BTserver(dataparams=self.datafeed_params,
                               min_episode_len=self.min_episode_len,
                               max_episode_time=self.max_episode_days,
                               strategy_class=self.custom_strategy_class,
                               network_address=self.network_address,
                               state_embed_dim=self.state_embed_dim,
                               state_dim_1=self.state_dim_1,
                               verbose=self.verbose)
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup
        time.sleep(1)
        self.log.info('Server started, pinging {} ...'.format(self.network_address))
        self.socket.send_pyobj('ping!')
        self.control_response = self.socket.recv_pyobj()
        self.log.info('Server seems ready with response: <{}>'.format(self.control_response))

    def _reset(self):
        """
        Implementation of OpenAI env.reset method.
        Rewinds backtrader server and starts new episode
        within randomly selected time period.
        """
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()
        # In case server is in 'episode mode'
        self.socket.send_pyobj('_done')
        self.control_response = self.socket.recv_pyobj()
        # Definetely, server now is  in 'control mode':
        self.socket.send_pyobj('_reset')
        self.control_response = self.socket.recv_pyobj()
        # Get initial episode response:
        self.step_response = self._step(0)
        # Check if state_space is as expected:
        try:
            assert self.step_response['state'].shape == self.observation_space.shape
        except:
            msg = ('\nState observation shape mismatch!\n' +
                   'Shape set by env: {},\n' +
                   'Shape returned by server: {}.\n' +
                   'Hint: Wrong get_state() parameters?').format(self.observation_space.shape,
                                                                 self.step_response['state'].shape)
            self.log.info(msg)
            self._close()
            raise AssertionError(msg)
        return self.step_response

    def _step(self, action):
        """
        Implementation of OpenAI env.step method.
        Relies on remote backtrader server for actual environment dynamics computing.
        """
        # Are YOU in The Actions List?
        assert self.action_space.contains(action)
        # Send action to backtrader engine, recieve response
        self.socket.send_pyobj(self.server_actions[action])
        self.step_response = self.socket.recv_pyobj()
        # DBG
        self.log.debug('Step(): recieved response {} as {}'.format(self.step_response, type(self.step_response)))
        return self.step_response

    def _close(self):
        """
        Stops BT server process
        """
        if not self.server:
            self.log.info('No server process found. Hint: Forgot to start?')
        else:
            if self.server.is_alive():
                if not self.socket.closed:
                    self.socket.send_pyobj('_done')
                    self.control_response = self.socket.recv_pyobj()
                    self.socket.send_pyobj('_stop')
                    self.control_response = self.socket.recv_pyobj()
                else:
                    self.server.terminate()
                    self.server.join()
            else:
                self.log.info('Server seems stopped already.')
            self.log.info('Server process exit code: {}'.format(self.server.exitcode))


############################## Server part ##############################

class TestDataLen(bt.Strategy):
    """
    Service strategy, used by <EpisodicDataFeed>.test_data_period() method
    """
    params = dict(start_date=None, end_date=None)

    def nextstart(self):
        self.p.start_date = self.data.datetime.date()

    def stop(self):
        self.p.end_date = self.data.datetime.date()


class EpisodicDataFeed():
    """ 
    BTfeeds class wrapper. Implements random episode sampling.
    Doesn't rely on pandas, works ok, but is pain-slow and needs rewriting.
    """

    def __init__(self, BTFeedDataClass):
        self.dataclass = BTFeedDataClass
        self.log = logging.getLogger('Epidode datafeed')

    def test_data_period(self, fromdate=None, todate=None):
        """
        Takes datafeed class and date/time constraints;
        returns actual number of  records and start/finish dates
        instantiated datafeed would contain under given constraints.
        """
        TestData = bt.Cerebro(stdstats=False)
        TestData.addstrategy(TestDataLen)
        data = self.dataclass(fromdate=fromdate, todate=todate)
        TestData.adddata(data)
        self.log.info('Init.  data range from {} to {}'.format(data.params.fromdate, data.params.todate))
        result = TestData.run()[0]
        self.log.info('Result data range from {} to {}'.format(result.p.start_date, result.p.end_date))
        return [result.data.close.buflen(),
                result.p.start_date,
                result.p.end_date, ]

    def sample_episode(self, min_len=2000, max_days=2):
        """
        For dataclass passed in, randomly samples
        <params.fromdate> and <params.todate> constraints;
        returns constrained datafeed, such as:
        datafeed number of records >= min_len,
        datafeed datetime period <= max_days.
        """
        max_days_stamp = 86400 * max_days
        while True:
            # Keep sampling random timestamps in datadeed first/last stamps range
            # until minimum  length condition is satisfied:

            # !!-->TODO: Can loop forever, if something is wrong with data, etc.
            # need sanity check: rise exeption after 100 retries ???

            self.log.info('Sampling episode...')
            rnd_start_timestamp = int(self.firststamp \
                                      + (self.laststamp - max_days_stamp
                                         - self.firststamp) * random.random())
            # Add maximum days:
            rnd_end_timestamp = rnd_start_timestamp + max_days_stamp
            # Convert to datetime:
            random_fromdate = datetime.datetime.fromtimestamp(rnd_start_timestamp)
            random_todate = datetime.datetime.fromtimestamp(rnd_end_timestamp)
            # Minimum length check:
            [num_records, _1, _2] = self.test_data_period(random_fromdate, random_todate)
            self.log.info('Episode length: {}'.format(num_records))

            if num_records >= min_len: break
        random_datafeed = self.dataclass(fromdate=random_fromdate,
                                         todate=random_todate,
                                         numrecords=num_records)
        self.log.info('Sample accepted.')
        return random_datafeed

    def measure(self):
        """
        Stores statistic for entire datafeed:
        total nuber of rows (records),
        date/times of first and last records (as timestamps).
        """
        self.log.info('Looking up entire datafeed. It may take some time.')
        [self.numrecords, self.firstdate, self.lastdate] = self.test_data_period(None, None)
        self.log.info('Total datafeed records: {}'.format(self.numrecords))
        self.firststamp = time.mktime(self.firstdate.timetuple())
        self.laststamp = time.mktime(self.lastdate.timetuple())


class BTserverStrategy(bt.Strategy):
    """
    Controls Environment inner dynamics.
    Custom State, Reward and Info computation logic can be implemented outside
    as subclass and passed to environment via <custom_strategy_class> parameter.
    To do this one should override get_state(), get_reward(), get_info(),
    set_datalines() methods (and do not forget locally import required packages).
    Since it is bt.Strategy subclass, see:
    https://www.backtrader.com/docu/strategy.html
    for more information.
    !!-->TODO: implement is_done as customisable method 
    """
    # Locally import required libraries. Call it as self.<name>!
    np = importlib.import_module('numpy')
    params = dict(
        socket=None,
        max_steps=0,
        state_dimension=10,
        state_dim_1=4,
        drawdown_call=10, )

    def __init__(self):
        self.p.log = logging.getLogger('WorkHorse')
        # A wacky way to define strategy 'minimum period' for proper embedded state composition:
        self.data.dim_sma = btind.SimpleMovingAverage(self.datas[0],
                                                      period=self.p.state_dimension)
        # Add custom Lines if any:
        self.set_datalines()

        # Housekeeping:
        self.iteration = 0
        self.action = 'hold'
        self.order = None
        self.order_message = '-'

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close.
        Any other custom data lines, indicators, etc.
        should be explicitly defined by overriding this method.
        Envoked once by Strategy.__init__().
        """
        pass

    def get_state(self):
        """
        Default state observation composer.
        Returns time-embedded environment state observation as [n,m] numpy matrix, where
        n - number of signal features,
        m - time-embedding length.
        One can override this method,
        defining nesessery calculations and return arbitrary shaped tensor.
        It's possible either to compute entire featurized environment state
        or just pass raw price data to RL algorithm featurizer module.
        Note1: 'data' referes to bt.startegy datafeed and should be treated as such.
        Datafeed Lines that are not default to BTserverStrategy should be explicitly defined in
        define_datalines().
        Note2: n is essentially == env.state_dim_1. 
        """
        self.state = self.np.row_stack((self.data.open.get(size=self.p.state_dimension),
                                        self.data.low.get(size=self.p.state_dimension),
                                        self.data.high.get(size=self.p.state_dimension),
                                        self.data.close.get(size=self.p.state_dimension),))

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
        which can be any string/object. Override by own taste.
        """
        self.info = ('Step: {}\nAgent action: {}\n' +
                     'Portfolio Value: {:.5f}\n' +
                     'Reward: {:.4f}\n{}\n' +
                     'Drawdown: {:.4f}\n' +
                     'Max.Drawdown: {:.4f}\n').format(self.iteration,
                                                      self.action,
                                                      self.stats.broker.value[0],
                                                      self.reward,
                                                      self.order_message,
                                                      self.stats.drawdown.drawdown[0],
                                                      self.stats.drawdown.maxdrawdown[0])

    def episode_stop(self):
        """
        Well, finishes current episode.
        """
        self.env.runstop()

    def notify_order(self, order):
        """
        Well...
        Just ripped from backtrader tutorial.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.order_message = 'BUY executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.order_message = 'SELL executed,\nPrice: {:.5f}, Cost: {:.4f}, Comm: {:.4f}'. \
                    format(order.executed.price,
                           order.executed.value,
                           order.executed.comm)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled]:
            self.order_message = 'Order Canceled'
        elif order.status in [order.Margin, order.Rejected]:
            self.order_message = 'Order Margin/Rejected'
        self.order = None

    def next(self):
        """
        Defines one step environment routine for server 'Episode mode'
        """
        # Housekeeping:
        is_done = False
        self.iteration += 1

        # Will it be last step of the episode?
        if self.iteration >= self.data.p.numrecords - self.p.state_dimension:
            is_done = True
            self.order_message = 'END OF DATA!'
        else:
            # All the strategy computing should be performed here as function of <self.action>,
            # this ensures async. server/client computations.
            # Note: that implies that action execution is delayed for 1 step.
            # <is_done> flag can also be rised here by trading logic, e.g. <OMG! We became too rich!>
            #
            if self.stats.drawdown.maxdrawdown[0] > self.p.drawdown_call:  # Any money left?
                is_done = True  # Trade No More
                self.order_message = 'DRAWDOWN CALL!'
            else:
                if self.action == 'hold' or self.order:
                    pass
                elif self.action == 'buy':
                    self.order = self.buy()
                    self.order_message = 'New BUY created; ' + self.order_message
                elif self.action == 'sell':
                    self.order = self.sell()
                    self.order_message = 'New SELL created; ' + self.order_message
                elif self.action == 'close':
                    self.order = self.close()
                    self.order_message = 'New CLOSE created; ' + self.order_message

                    # Gather response:
        self.get_state()
        self.get_reward()
        self.get_info()  # should be on last place!

        # Receive action from outer world:
        next_action = self.p.socket.recv_pyobj()
        self.p.log.debug('Server recieved: {} as: {}'.format(next_action, type(next_action)))
        if next_action == '_done': is_done = True  # = client calls env.reset()

        # Send response:
        self.p.socket.send_pyobj({'state': self.state,
                                  'reward': self.reward,
                                  'done': is_done,
                                  'info': self.info})
        # Housekeeping:
        self.action = next_action  # carry action to be executed by next step
        self.order_message = '-'  # clear order message

        # Maybe initiate fallback to Control Mode?
        if is_done:
            self.close()
            self.episode_stop()


class BTserver(multiprocessing.Process):
    """
    Backtrader server.
    Control signals:
    IN:
    '_reset' - rewinds backtrader engine and runs new episode;
    '_stop' - server shut-down.
    OUT:
    info: <string message> - reports current server status.
    Whithin-episode signals:
    IN:
    {'buy', 'sell', 'hold', 'close', '_done'} - actions;
                                           *'_done' - stops current episode.
    OUT:
    response - <dict>: observation - observation of the current environment state,
                                     could be any tensor; default is:
                                     [4,m] array of <fl32>, where:
                                     m - num. of last datafeed values,
                                     4 - num. of data features (Lines);
                       reward - current portfolio statistics for environment reward estimation;
                       done - episode termination flag;
                       info - auxiliary diagnostic information, if any.

    Parameters:
    dataparms - CSV file name and parsing parameters;
    max_steps - <int>, maximum episode length;
    network_address - <str>, network address to bind to;
    cerebro_params - <dict>: trading engine-specific parameters, excl. datafeed 
    """

    def __init__(self,
                 dataparams,
                 min_episode_len,
                 max_episode_time,
                 network_address,
                 strategy_class=None,
                 state_embed_dim=10,
                 state_dim_1=4,
                 verbose=False):
        """
        Configures BT server instance.
        """
        super(BTserver, self).__init__()
        self.dataparams = dataparams
        self.min_episode_len = min_episode_len
        self.max_episode_time = max_episode_time
        self.network_address = network_address
        self.state_embed_dim = state_embed_dim
        self.state_dim_1 = state_dim_1
        self.verbose = verbose

        # Use default strategy class in none has been passed in:
        if not strategy_class:
            self.strategy_class = BTserverStrategy
        else:
            self.strategy_class = strategy_class

        # Define datafeed:
        class CSVData(btfeeds.GenericCSVData):
            """Backtrader CSV datafeed class"""
            params = self.dataparams

        self.data = EpisodicDataFeed(CSVData)

    def run(self):
        """
        Server process execution body. This method is envoked by env._start_server().
        """
        # Verbosity control:
        if self.verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        log = logging.getLogger('BT_server main')
        self.process = multiprocessing.current_process()
        log.info('Server process PID: {}'.format(self.process.pid))
        wonderful_results = []
        some_important_statisitc = []

        # Set up a comm. channel for server as zmq socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        # Lookup datafeed:
        self.data.measure()

        # Server 'Control Mode' loop:
        for episode_number in itertools.count(1):
            # Stuck here until 'reset' or 'stop':
            while True:
                service_input = socket.recv_pyobj()
                # Check if it's time to exit:
                if service_input == '_stop':
                    # Release comm channel, gather statistic and exit:
                    # TODO: Gather and somehow pass over global statistics
                    # Server shutdown logic:
                    some_important_statisitc = True
                    message = 'Server is exiting.'
                    log.info(message)
                    socket.send_pyobj(message)  # pairs '_stop' input
                    socket.close()
                    context.destroy()
                    return wonderful_results, some_important_statisitc
                    # And where do you think you actually return it, hah?  <-- TODO: dump stats to file or something
                if service_input == '_reset':
                    message = 'Starting new episode'
                    log.info(message)
                    socket.send_pyobj(message)  # pairs '_reset'
                    break
                message = 'Server control mode, send <_reset> or <_stop>.'
                # log.info(message)
                socket.send_pyobj(message)  # pairs any other input

            # Got '_reset' signal, prepare Cerebro instance and enter 'Episode Mode':
            cerebro = bt.Cerebro(stdstats=False)
            cerebro.addstrategy(self.strategy_class,
                                socket=socket,
                                state_dimension=self.state_embed_dim,
                                state_dim_1=self.state_dim_1)
            cerebro.broker.setcash(10.0)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addobserver(bt.observers.DrawDown)
            cerebro.addsizer(bt.sizers.SizerFix, stake=10)

            cerebro.adddata(self.data.sample_episode(self.min_episode_len, self.max_episode_time))
            # log.info('Starting Portfolio Value: {:.4f}'.format(cerebro.broker.getvalue()))
            wonderful_results = cerebro.run(stdstats=True)
            # log.info('Final Portfolio Value: {:.4f}'.format(cerebro.broker.getvalue()))

        # Just in case -- we actually shouldnt get there except by some error
        return wonderful_results, some_important_statisitc

