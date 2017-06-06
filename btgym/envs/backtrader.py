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
import time
import zmq
import os

import gym
from gym import error, spaces
#from gym import utils
#from gym.utils import seeding, closer

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

from btgym.server import BTserver
from btgym.strategy import BTserverStrategy

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

