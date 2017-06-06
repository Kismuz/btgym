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

from btgym.server import BTserver
from btgym.strategy import BTserverStrategy

############################## Environment part ##############################

class BacktraderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_filename,  # server arg, required, CSV data filename
                 datafeed_params=None,  # server arg/ if None - default CSV parsing params will be used
                 port=5500,  # server arg/ port to use
                 min_episode_len=2000,  # cerebro arg/ minimum episode number of data records (rows)
                 max_episode_days=2,  # cerebro arg/ maximum episode time in days
                 cerebro=None,   # server arg/ bt.cerbro class for server to execute,
                                 # if None - default strategy will be used
                 state_dim_time=10,  # environment/cerebro.strategy arg/ state observation time-embedding dimensionality
                                     # get overriden if cerebro arg is not None
                 state_dim_0=4,  # environment/cerebro.strategy arg/ state observation feature dimensionality,
                                 # get overriden if cerebro arg is not None
                 portfolio_actions=('hold', 'buy', 'sell', 'close'),  # environment/[strategy] arg/ agent actions
                 verbose=False, ):  # environment/server arg

        # Check datafile existence:
        if not os.path.isfile(data_filename):
            raise FileNotFoundError('Datafeed not found: ' + data_filename)

        # Verbosity control:
        self.log = logging.getLogger('Env')
        if verbose:
            if verbose == 2:
                logging.getLogger().setLevel(logging.DEBUG)
            else:
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
                info='1 min FX generic ASCII, www.HistData.com', )
        else:
            self.datafeed_params = datafeed_params
            self.datafeed_params['dataname'] = data_filename
        """
        Default configuration for Backtrader computational engine (cerebro).
        Executed only if no bt.Cerebro custom subclass has been given:
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

        # Episode related params, added to any cerebro class:
        self.cerebro.min_episode_len = min_episode_len
        self.cerebro.max_episode_days = max_episode_days

        # Server process/network parameters:
        self.server = None
        self.port = port
        self.network_address = 'tcp://127.0.0.1:{}'.format(port)

        # Infer env. observation space from cerebro strategy parameters,
        # default is 2d matrix, values in [0,10]. Override if needed:
        self.observation_space = spaces.Box(low=0.0,
                                            high=10.0,
                                            shape=(self.cerebro.strats[0][0][2]['state_dim_0'],
                                                   self.cerebro.strats[0][0][2]['state_dim_time']))
        self.log.debug('OBS. SHAPE: {}'.format(self.observation_space.shape))

        # Action space and corresponding server messages:
        self.action_space = spaces.Discrete(len(portfolio_actions))
        self.server_actions = portfolio_actions + ('_done', '_reset', '_stop','_getstat')

        # Set up client channel:
        # First, kill any process using server port:
        # TODO: sort of overkill?
        cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.port)
        os.system(cmd)
        # Summon ZMQ:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.network_address)

        # Finally:
        self.log.info('Environment is ready.')

    def _start_server(self):
        """
        Configures backtrader REQ/REP server instance and starts server process.
        """
        self.server = BTserver(dataparams=self.datafeed_params,
                               cerebro=self.cerebro,
                               network_address=self.network_address,
                               verbose=self.verbose)
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup
        time.sleep(1)
        self.log.info('Server started, pinging {} ...'.format(self.network_address))
        self.socket.send_pyobj('ping!')
        self.server_response = self.socket.recv_pyobj()
        self.log.info('Server seems ready with response: <{}>'.format(self.server_response))

    def _reset(self):
        """
        Implementation of OpenAI Gym env.reset method.
        'Rewinds' backtrader server and starts new episode
        within randomly selected time period.
        """
        # Server process check:
        if not self.server or not self.server.is_alive():
            self.log.info('No running server found, starting...')
            self._start_server()

        if self._force_control_mode():
            self.socket.send_pyobj('_reset')
            self.server_response = self.socket.recv_pyobj()

            # Get initial episode response:
            self.server_response = self._step(0)

            # Check if state_space is as expected:
            try:
                assert self.server_response['state'].shape == self.observation_space.shape
            except:
                msg = ('\nState observation shape mismatch!\n' +
                       'Shape set by env: {},\n' +
                       'Shape returned by server: {}.\n' +
                       'Hint: Wrong get_state() parameters?').format(self.observation_space.shape,
                                                                     self.server_response['state'].shape)
                self.log.info(msg)
                self._stop_server()
                raise AssertionError(msg)
            return self.server_response
        else:
            msg = 'Something went wrong. env.reset() cant get response from server.'
            self.log.info(msg)
            raise ChildProcessError(msg)

    def _step(self, action):
        """
        Implementation of OpenAI Gym env.step method.
        Relies on remote backtrader server for actual environment dynamics computing.
        """
        # Are you in the list?
        assert self.action_space.contains(action)
        # Send action to backtrader engine, recieve response
        self.socket.send_pyobj(self.server_actions[action])
        self.server_response = self.socket.recv_pyobj()
        self.log.debug('Step(): recieved response {} as {}'.format(self.server_response, type(self.server_response)))
        return self.server_response

    def _close(self):
        """
        [kind of] Implementation of OpenAI Gym env.close method.
        Puts server in Control Mode
        """
        _ = self._force_control_mode()
        # maybe TODO something

    def _force_control_mode(self):
        """
        Puts BT server to control mode.
        """
        # Is there any server process?
        if not self.server or not self.server.is_alive():
            msg = 'No running server found.'
            self.log.info(msg)
            self.server_response = msg
            return False
        else:
            self.server_response = 'NONE'
            attempt = 0
            while not 'CONTROL' in self.server_response:
                self.socket.send_pyobj('_done')
                self.server_response = self.socket.recv_pyobj()
                attempt += 1
                self.log.debug('FORCE CONTROL MODE attempt: {}.\nResponse: {}'.format(attempt, self.server_response))
            return True

    def get_stat(self):
        """
        Returns last episode statistics.
        Note: when evoked, forces running episode to terminate.
        """
        if self._force_control_mode():
            self.socket.send_pyobj('_getstat')
            return self.socket.recv_pyobj()
        else:
            return self.server_response

    def _stop_server(self):
        """
        Stops BT server process.
        """
        if not self.server:
            self.log.info('No server process found. Hint: Forgot to start?')
        else:
            if self._force_control_mode():
                if not self.socket.closed:
                    self.socket.send_pyobj('_stop')
                    self.server_response = self.socket.recv_pyobj()
                else:
                    self.server.terminate()
                    self.server.join()
            else:
                self.log.info('Server seems stopped already.')
            self.log.info('Server process exit code: {}'.format(self.server.exitcode))
