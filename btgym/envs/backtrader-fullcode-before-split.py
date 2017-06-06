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
import copy

import os

import gym
from gym import error, spaces
#from gym import utils
#from gym.utils import seeding, closer

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

############################## Environment part ##############################

class BacktraderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_filename=None, #TODO: server/Data params
                 datafeed_params=None, #TODO: server/Data params
                 port=5500, #TODO: server/Communicator params
                 min_episode_len=2000, #TODO: server/Data params
                 max_episode_days=2, #TODO: server/Data params
                 cerebro=None, 
                 state_dim_time=10, # TODO: RL PARAM
                 state_dim_0=4, # TODO: RL PARAM
                 # TODO:  <--- encapsulate by adding: stats, brokers, analyzers, sizers etc. PASS via CEREBRO_CLASS
                 portfolio_actions=['hold', 'buy', 'sell', 'close'], # TODO: server/Communicator params
                 verbose=False, ):

        # Check datafile existence:
        if not os.path.isfile(data_filename):
            raise FileNotFoundError('Datafeed not found: ' + data_filename)

        # Verbosity control:
        self.log = logging.getLogger('Env')
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        self.verbose = verbose

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

        """
        Configure default Backtrader computational engine (cerebro).
        Executed only if bt.Cerebro custom subclass has been passed to environment.
        """
        if not cerebro:
            self.cerebro = bt.Cerebro()
            self.cerebro.addstrategy(BTserverStrategy,
                                     state_dim_time=state_dim_time,
                                     state_dim_0=state_dim_0)
            self.cerebro.broker.setcash(10.0)
            self.cerebro.broker.setcommission(commission=0.001)
            self.cerebro.addobserver(bt.observers.DrawDown)
            self.cerebro.addsizer(bt.sizers.SizerFix, stake=10)
        else:
            self.cerebro = cerebro

        # Server/network parameters:
        self.server = None
        self.port = port
        self.network_address = 'tcp://127.0.0.1:{}'.format(port)

        # Episode data related parameters:
        self.min_episode_len = min_episode_len
        self.max_episode_days = max_episode_days

        # Infer observation space from cerebro parameters:
        # Observation space:
        self.state_dim_time = state_dim_time
        self.state_dim_0 = state_dim_0
        # 2dim matrix in [0,10]. Override if needed:
        self.observation_space = spaces.Box(low=0.0,
                                            high=10.0,
                                            shape=(self.state_dim_0,
                                                   self.state_dim_time))

        # Action space and corresponding server messages:
        self.action_space = spaces.Discrete(len(portfolio_actions))
        self.server_actions = portfolio_actions + ['_done', '_reset', '_stop','_getstat']

        # Set client channel:
        # first, kill any process using server port:
        # cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.port)
        # os.system(cmd)
        # Summon ZMQ:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.network_address)

        # Finally:
        self.log.info('Environment is ready...')

    def _start_server(self):
        """
        Configures backtrader REQ/REP server instance and starts server process.
        """
        self.server = BTserver(dataparams=self.datafeed_params,
                               min_episode_len=self.min_episode_len,
                               max_episode_time=self.max_episode_days,
                               cerebro=self.cerebro,
                               network_address=self.network_address,
                               state_dim_time=self.state_dim_time,
                               state_dim_0=self.state_dim_0,
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
        Implementation of OpenAI Gym env.reset method.
        Rewinds backtrader server and starts new episode
        within randomly selected time period.
        """
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()
        # Paranoid 'episode mode' check:
        self.control_response = '---'
        while not 'Control mode' in self.control_response:
            self.socket.send_pyobj('_done')
            self.control_response = self.socket.recv_pyobj()
        # Now, it is in 'control mode':
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
        Implementation of OpenAI Gym env.step method.
        Relies on remote backtrader server for actual environment dynamics computing.
        """
        # Are you in the list?
        assert self.action_space.contains(action)
        # Send action to backtrader engine, recieve response
        self.socket.send_pyobj(self.server_actions[action])
        self.step_response = self.socket.recv_pyobj()
        # DBG
        self.log.debug('Step(): recieved response {} as {}'.format(self.step_response, type(self.step_response)))
        return self.step_response

    def _close(self):
        """
        Implementation of OpenAI Gym env.close method.
        Puts server in Control Mode
        """
        # Paranoid close:
        self.control_response = '---'
        while not 'Control mode' in self.control_response:
            self.socket.send_pyobj('_done')
            self.control_response = self.socket.recv_pyobj()

    def get_stat(self):
        """
        Returns last episode statistics.
        Note: when evoked, forces running (if any) episode to terminate.
        """
        # If there's server running?
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found')
            return None
        # Paranoid 'episode mode' check:
        self.control_response = '---'
        attempt = 0
        while not 'Control mode' in self.control_response:
            self.socket.send_pyobj('_done')
            self.control_response = self.socket.recv_pyobj()
            attempt +=1
            self.log.debug('GET_STAT ATTEMPT: {}\nRESPONSE: '.format (attempt, self.control_response))
        # Now, got that control mode:
        self.socket.send_pyobj('_getstat')
        return self.socket.recv_pyobj()

    def _stop_server(self):
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


############################## BT Server in-episode comm. method ##############################

class _EpisodeComm(bt.Analyzer):
    """
    Performs client-server REQ/REP communication while in episode mode.
    No, as part of core server operational logic, should not be explicitly called/edited by user.
    Yes, it analyzes nothing.
    """
    log = None
    socket = None
    response = None
    #params = dict(
    #    socket = None,
    #    log = None,)

    def __init__(self):
        """
        Just get log and ZMQ socket from parent.
        """
        self.log = self.strategy.env._log
        self.socket = self.strategy.env._socket

    def prenext(self):
        pass

    def stop(self):
        pass

    def next(self):
        """
        Actual env.step() communication is here.
        """
        # Receive action from outer world:
        self.strategy.action = self.socket.recv_pyobj()
        self.log.debug('COMM recieved: {}'.format(self.strategy.action))
        self.response = {'state': self.strategy.state,
                         'reward': self.strategy.reward,
                         'done': self.strategy.is_done,
                         'info': self.strategy.info}
        # Send response:
        self.socket.send_pyobj(self.response)
        self.log.debug('COMM sent: {}//{}'.format(self.response['done'], self.response['info']))


############################## Base BTServer Strategy Class ##############################

class BTserverStrategy(bt.Strategy):
    """
    Controls Environment inner dynamics and backtesting logic.
    Any State, Reward and Info computation logic can be implemented by
    subclassing BTserverStrategy and overriding at least get_state(), get_reward(), get_info(),
    set_datalines() methods (and not forgetting locally import required packages!).
    One can always go deeper and override __init__ () and next() methods for desired
    server cerebro engine behaviour, including order execution etc.
    Since it is bt.Strategy subclass, see:
    https://www.backtrader.com/docu/strategy.html
    for more information.
    """
    # Locally import required libraries (call it as self.<lib-name>):
    importlib = __import__('importlib') # import importer
    np = importlib.import_module('numpy')
    # Set-list:
    log = None
    state = None
    reward = None
    info ='_'
    is_done = False
    iteration = 0
    action = 'hold'
    order = None
    broker_message = '-'
    params = dict(state_dim_time=10, # state time embedding dimension (just convention)
                  state_dim_0=4, # one can add dim_1, dim_2, ... if needed; should match env.observation_space
                  drawdown_call=20, ) # simplest condition to exit

    def __init__(self):
        # Inherit logger from cerebro:
        self.log = self.env._log

        # A wacky way to define strategy 'minimum period'
        # for proper time-embedded state composition:
        self.data.dim_sma = btind.SimpleMovingAverage(self.datas[0],
                                                      period=self.p.state_dim_time)
        # Add custom Lines if any (just a wrapper):
        self.set_datalines()

    def set_datalines(self):
        """
        Default datalines are: Open, Low, High, Close.
        Any other custom data lines, indicators, etc.
        should be explicitly defined by overriding this method.
        evoked once by Strategy.__init__().
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
        Note1: 'data' referes to bt.startegy datafeed and should be treated as such.
        Datafeed Lines that are not default to BTserverStrategy should be explicitly defined in
        define_datalines().
        Note2: 'n' is essentially == env.state_dim_0.
        """
        self.state = self.np.row_stack((self.data.open.get(size=self.p.state_dim_time),
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
                     '{}\n' + # Order message here
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
        if self.iteration >= self.data.p.numrecords - self.p.state_dim_time:
            self.broker_message = 'END OF DATA!'
        elif self.action =='_done':
            self.broker_message = '_DONE SIGNAL RECEIVED'
        # Any money left?:
        elif  self.stats.drawdown.maxdrawdown[0] > self.p.drawdown_call:
            self.broker_message = 'DRAWDOWN CALL!'
        #...............
        # Finally, it seems ok to continue:
        else:
            self.is_done = False
            return
        # Or else, initiate fallback to Control Mode; still executes strategy cycle once:
        self.log.debug('RUNSTOP() evoked with {}'.format(self.broker_message))
        self.env.runstop()

    def notify_order(self, order):
        """
        Just ripped from backtrader tutorial.
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
        nvoke following methods:
                self.get_done() - should be early in code,
                ....
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
            # All the strategy computations should be performed here as function of <self.action>,
            # this ensures async. server/client computations.
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
        else: # time to leave:
            self.close()
            
        # Gather response:
        self.get_state()
        self.get_reward()
        #self.get_done()
        self.get_info() 

        # Somewhere at this point, _EpisodeComm() is exchanging information with environment wrapper,
        # obtaining <self.action> and sending <state,reward,done,info>... Never mind.

############################## Episodic Datafeed Class ##############################

class TestDataLen(bt.Strategy):
    """
    Service strategy, used by <EpisodicDataFeed>.test_data_period() method.
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
            # need sanity check: e.g. rise exeption after 100 retries ???

            self.log.info('Sampling episode data:')
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
        total number of rows (records),
        date/times of first and last records (as timestamps).
        """
        self.log.info('Looking up entire datafeed. It may take some time.')
        [self.numrecords, self.firstdate, self.lastdate] = self.test_data_period(None, None)
        self.log.info('Total datafeed records: {}'.format(self.numrecords))
        self.firststamp = time.mktime(self.firstdate.timetuple())
        self.laststamp = time.mktime(self.lastdate.timetuple())

##############################  BT Server Main part ##############################

class BTserver(multiprocessing.Process):
    """
    Backtrader server class.
    Control signals:
    IN:
    '_reset' - rewinds backtrader engine and runs new episode;
    '_getstat' - retrieve last run episode results and statistics;
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
                 cerebro=None,
                 state_dim_time=10,
                 state_dim_0=4,
                 verbose=False):
        """
        Configures BT server instance.
        """
        super(BTserver, self).__init__()
        self.dataparams = dataparams
        self.min_episode_len = min_episode_len
        self.max_episode_time = max_episode_time
        self.network_address = network_address
        self.state_dim_time = state_dim_time
        self.state_dim_0 = state_dim_0
        self.verbose = verbose

        # Cerebro class to execute:
        if not cerebro:
            raise AssertionError('Server has not recieved any bt.cerebro() class. Nothing to run!')
        else:
            self.cerebro = cerebro

        # Define datafeed:
        class CSVData(btfeeds.GenericCSVData):
            """Backtrader CSV datafeed class"""
            params = self.dataparams
        self.data = EpisodicDataFeed(CSVData)

    def run(self):
        """
        Server process execution body. This method is evoked by env._start_server().
        """
        # Verbosity control:
        if self.verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        log = logging.getLogger('BT_server')

        self.process = multiprocessing.current_process()
        log.info('Server process PID: {}'.format(self.process.pid))

        # Housekeeping:
        cerebro_result = 'No runs has been made.'

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
                log.debug('Control mode: recieved <{}>'.format(service_input))
                # Check if it's time to exit:
                if service_input == '_stop':
                    # Server shutdown logic:
                    # send last run statistic, release comm channel and exit:
                    log.info('Server is exiting.')
                    socket.send_pyobj(cerebro_result)
                    socket.close()
                    context.destroy()
                    return None

                elif service_input == '_reset':
                    message = 'Starting episode.'
                    log.info(message)
                    socket.send_pyobj(message)  # pairs '_reset'
                    break

                elif service_input == '_getstat':
                    socket.send_pyobj(episode_result)
                    log.info('Episode statistic sent.')

                else: # ignore any other input
                    message = 'Control mode, send <_reset>, <_getstat> or <_stop>.'
                    log.debug(message)
                    socket.send_pyobj(message)  # pairs any other input

            # Got '_reset' signal, prepare Cerebro subclass and run episode:
            cerebro = copy.deepcopy(self.cerebro)

            cerebro._socket = socket
            cerebro._log = log

            # Add communication ability:
            cerebro.addanalyzer(_EpisodeComm,
                                _name='communicator',)

            # Add random episode data:
            cerebro.adddata(self.data.sample_episode(self.min_episode_len, self.max_episode_time))

            # Finally:
            episode = cerebro.run(stdstats=True)[0]
            log.info('Episode finished.')

            # Get statistics:
            episode_result = dict(episode = episode_number,)
                                  #stats = episode.stats,
                                  #analyzers = episode.analyzers,
            log.debug('ANALYZERS: {}'.format(len(episode.analyzers)))
            log.debug('DATADEEDS: {}'.format(len(episode.datas)))

        # Just in case -- we actually shouldnt get there except by some error:
        return None

