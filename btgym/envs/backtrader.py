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

from btgym import BTgymServer, BTgymStrategy, BTgymDataset

############################## OpenAI Gym Environment  ##############################

class BTgymEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for Backtrader backtesting/trading library.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.dataset = None  # BTgymDataset instance,

        self.engine = None  # bt.Cerbro subclass for server to execute.

        self.strategy = None  # base strategy to use if no <strategy> kwarg been passed.

        # Default parameters:
        self.params = dict(
            dataset = dict(
                # Dataset parameters:
                filename=None,  # Source CSV data file;
                # Episode data params:
                start_weekdays=[0, 1, 2, ],  # Only weekdays from the list will be used for episode start.
                start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).
                episode_len_days=1,  # Maximum episode time duration in d:h:m.
                episode_len_hours=23,
                episode_len_minutes=55,
                time_gap_days=0,  # Maximum data time gap allowed within sample in d:h.
                time_gap_hours=5,  # If set < 1 day, samples containing weekends and holidays gaps will be rejected.
            ),
            engine = dict(
                # Backtrader engine parameters:
                start_cash=10.0,  # initial trading capital.
                broker_commission=0.001,  # trade execution commission, default is 0.1% of operation value.
                fixed_stake=10,  # single trade stake is fixed type by def.
            ),
            strategy = dict(
                # Strategy related parameters:
                state_shape=(4, 10),  # observation state shape, by convention last dimension is time embedding;
                # one can define any shape; match env.observation_space.shape.
                state_low=None,  # observation space state min/max values,
                state_high=None,  # if set to None - absolute min/max values from BTgymDataset will be used.
                drawdown_call=90,  # episode maximum drawdown threshold, default is 90% of initial value.
                portfolio_actions=('hold', 'buy', 'sell', 'close'),  # environment/[strategy] arg/ agent actions,
                # should consist with BTgymStrategy order execution logic;
                # defaults are: 0 - 'do nothing', 1 - 'buy', 2 - 'sell', 3 - 'close position'.
            ),
            other = dict(
                # Other:
                port=5500,  # network port to use.
                network_address='tcp://127.0.0.1:',   # using localhost.
                verbose=0,  # verbosity mode: 0 - silent, 1 - info level, 2 - debugging level
            ),
        )
        # Here we'll sort and store our kwargs:
        self.kwargs = dict()

        '''
        Environment kwargs applying logic:
    
        if <engine> kwarg is given:
            do not use default enfine and startegy parameters;
            ignore <startegy> kwarg and all startegy and engine-related kwargs.
        
        else (no <engine>):
            use default engine parameters;
            if any engine-related kwarg is given:
                override corresponding default parameter;
            
            if <strategy> is given:
                do not use default strategy parameters;
                if  any strategy related kwarg is given:
                    override corresponding strategy parameter;
                
            else (no <strategy>):
                use default strategy parameters;
                if  any strategy related kwarg is given:
                    override corresponding strategy parameter;
        
        if <dataset> kwarg is given:
            do not use default dataset parameters;
            if  any dataset related kwarg is given:
                    override corresponding dataset parameter;
                    
        else (no <dataset>):
            use default dataset parameters;
                if  any dataset related kwarg is given:
                    override corresponding dataset parameter;
        
        If any <other> kwarg is given:
            override corr. default parameter.
        '''
        # Sort kwargs out:
        for name, subset in self.params.items():
            self.kwargs[name] = dict()
            for key, value in kwargs.items():
                if key in subset:
                    self.kwargs[name][key] = value

        # Set <other> env attributes:
        self.params['other'].update(self.kwargs['other'])

        # Verbosity control:
        self.log = logging.getLogger('Env')
        if self.params['other']['verbose']:

            if self.params['other']['verbose'] == 2:
                logging.getLogger().setLevel(logging.DEBUG)

            else:
                logging.getLogger().setLevel(logging.INFO)

        else:
            logging.getLogger().setLevel(logging.ERROR)

        # Server process/network parameters:
        self.server = None
        self.context = None
        self.socket = None
        self.params['other']['network_address'] += str(self.params['other']['port'])

        # Dataset preparation:
        #
        if 'dataset' in kwargs:
            # If BTgymDataset instance has been passed:
            print(self.kwargs['dataset'])
            self.dataset = kwargs['dataset']
            # Update dataset parameters with kwargs:
            for key, value in self.kwargs['dataset'].items():
                if key in dir(self.dataset):
                    setattr(self.dataset, key, value)
            # Cleanup:
            msg = 'custom Dataset class used'
            self.params['dataset'] = dict(info=msg)

        else:
            # If no BTgymDataset has been passed,
            # Make default dataset with given CSV file:
            try:
                os.path.isfile(str(self.kwargs['dataset']['filename']))

            except:
                raise FileNotFoundError('Dataset source data file not specified/found')

            # Use kwargs on top of defaults:
            self.params['dataset'].update(self.kwargs['dataset'])
            self.dataset = BTgymDataset(**self.params['dataset'])
            msg = 'base Dataset class used'
        # Append logging:
        self.dataset.log = self.log

        self.log.info(msg)

        # Engine preparation:
        if 'engine' in kwargs:
            # If full-blown bt.Cerebro() subclass has been passed:
            self.engine = kwargs['engine']
            # Cleanup:
            msg = 'custom Cerebro engine used'
            self.params['engine'] = dict(info=msg)
            self.params['strategy'] = dict(info=msg)
            # ...and it's done.

        # Note: either way, bt.observers.DrawDown observer [and logger] will be added to any BTgymStrategy instance
        # by BTgymServer process at runtime.

        else:
            # Default configuration for Backtrader computational engine (Cerebro),
            # if no bt.Cerebro() custom subclass has been passed:
            self.engine = bt.Cerebro()
            msg = 'base Cerebro engine used'

            # First, set strategy configuration:
            if 'strategy' in kwargs:
                # If custom strategy has been passed:
                self.strategy = kwargs['strategy']
                # Add it along with kwargs (ignore defaults):
                self.engine.addstrategy(self.strategy,**self.kwargs['strategy'])
                # Cleanup:
                msg2 = 'custom Strategy class used'
                self.params['startegy'] = dict(info=msg2)

            else:
                # Base class strategy, using kwargs on top of defaults:
                self.strategy = BTgymStrategy
                self.params['strategy'].update(self.kwargs['strategy'])
                self.engine.addstrategy(self.strategy, **self.params['strategy'])
                msg2 = 'base Strategy class used'

            msg += ', ' + msg2

            # Second, set Cerebro-level configuration:
            self.engine.broker.setcash(self.params['engine']['start_cash'])
            self.engine.broker.setcommission(self.params['engine']['broker_commission'])
            self.engine.addsizer(bt.sizers.SizerFix, stake=self.params['engine']['fixed_stake'],)

        self.log.info(msg)

        # DEBUG:
        #print('HIGH:', self.params['strategy']['state_high'] )
        #print('LOW:', self.params['strategy']['state_low'])

        # Define observation space shape, minimum / maximum values and agent action space.


        # Retrieve values from configured engine:
        for key in ['state_shape', 'state_low', 'state_high', 'portfolio_actions',]:
            try:
                # Try to pull it from strategy 'passed params':
                self.params['strategy'][key] = self.engine.strats[0][0][2][key]

            except:
                # else dig it from strategy defaults:
                for t_key, t_value in self.engine.strats[0][0][0].params._gettuple():
                    if t_key == key:
                        self.params['strategy'][t_key] = t_value

        # For min/max, if not been set explicitly,
        # the only sensible way is to infer from raw Dataset price values:
        if self.engine.strats[0][0][2]['state_low'] == None or \
            self.engine.strats[0][0][2]['state_high'] == None:

            # Get dataset statistic:
            self.dataset_stat = self.dataset.describe()

            # Exclude 'volume' from columns we count:
            data_columns = list(self.dataset.names)
            data_columns.remove('volume')

            # Override with absolute price min and max values:
            self.params['strategy']['state_low'] =\
                self.engine.strats[0][0][2]['state_low'] =\
                self.dataset_stat.loc['min',data_columns].min()

            self.params['strategy']['state_high'] =\
                self.engine.strats[0][0][2]['state_high'] =\
                self.dataset_stat.loc['max', data_columns].max()

            self.log.info('Inferring obs. space high/low form dataset: {:.6f} / {:.6f}.'.
                          format(self.params['strategy']['state_low'],
                                 self.params['strategy']['state_high']))

        # Set observation space shape from engine/strategy parameters:
        self.observation_space = spaces.Box(low=self.params['strategy']['state_low'],
                                            high=self.params['strategy']['state_high'],
                                            shape=self.params['strategy']['state_shape'],)

        self.log.debug('Obs. shape: {}'.format(self.observation_space.shape))
        self.log.debug('Obs. min:\n{}\nmax:\n{}'.format(self.observation_space.low, self.observation_space.high))

        # Set action space and corresponding server messages:
        self.action_space = spaces.Discrete(len(self.params['strategy']['portfolio_actions']))
        self.server_actions = self.params['strategy']['portfolio_actions'] + ('_done', '_reset', '_stop','_getstat')


        # Finally:
        self.log.info('Environment is ready.')

    def _start_server(self):
        """
        Configures backtrader REQ/REP server instance and starts server process.
        """
        # Ensure network resources:
        # 1. Release client-side, if any:
        if self.context:
            self.context.destroy()
            self.socket = None

        # 2. Kill any process using server port:
        cmd = "kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(self.params['other']['port'])
        os.system(cmd)

        # Set up client channel:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.params['other']['network_address'])

        # Configure and start server:
        self.server = BTgymServer(dataset=self.dataset,
                                  cerebro=self.engine,
                                  network_address=self.params['other']['network_address'],
                                  log=self.log)
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup
        time.sleep(1)

        self.log.info('Server started, pinging {} ...'.format(self.params['other']['network_address']))
        self.socket.send_pyobj('ping!')
        self.server_response = self.socket.recv_pyobj()
        self.log.info('Server seems ready with response: <{}>'.format(self.server_response))

    def _stop_server(self):
        """
        Stops BT server process, releases network resources.
        """

        if not self.server:
            self.log.info('No server process found.')

        else:

            if self._force_control_mode():
                # In case server is running and client side is ok:
                self.socket.send_pyobj('_stop')
                self.server_response = self.socket.recv_pyobj()

            else:
                self.server.terminate()
                self.server.join()
                self.server_response = 'Server process terminated.'

            self.log.info('{} Exit code: {}'.format(self.server_response,
                                                    self.server.exitcode))

        # Release client-side, if any:
        if self.context:
            self.context.destroy()


    def _force_control_mode(self):
        """
        Puts BT server to control mode.
        """
        # Check is there any faults with server process and connection?
        network_error = [
            (not self.server or not self.server.is_alive(), 'No running server found.'),
            (not self.context or self.context.closed, 'No network connection found.'),
        ]
        for (err, msg) in network_error:
            if err:
                self.log.info(msg)
                self.server_response = msg
                return False

        else:
            # If everything works, insist to go 'control':
            self.server_response = 'NONE'
            attempt = 0

            while not 'CONTROL_MODE' in str(self.server_response):
                self.socket.send_pyobj('_done')
                self.server_response = self.socket.recv_pyobj()
                attempt += 1
                self.log.debug('FORCE CONTROL MODE attempt: {}.\nResponse: {}'.format(attempt, self.server_response))

            return True

    def _reset(self,
               state_only=True): # By default, returns only initial state observation (Gym convention).
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
                assert self.server_response[0].shape == self.observation_space.shape

            except:
                msg = ('\nState observation shape mismatch!\n' +
                       'Shape set by env: {},\n' +
                       'Shape returned by server: {}.\n' +
                       'Hint: Wrong get_state() parameters?').format(self.observation_space.shape,
                                                                     self.server_response[0].shape)
                self.log.info(msg)
                self._stop_server()
                raise AssertionError(msg)

            if state_only:
                return self.server_response[0]
            else:
                return self.server_response

        else:
            msg = 'Something went wrong. env.reset() can not get response from server.'
            self.log.info(msg)
            raise ChildProcessError(msg)

    def _step(self, action):
        """
        Implementation of OpenAI Gym env.step method.
        Relies on remote backtrader server for actual environment dynamics computing.
        """

        # Are you in the list and ready to go?
        try:
            assert self.action_space.contains(action) and (self.socket and not self.socket.closed)

        except:
            msg = ('\nAt least one of these occurred:\n' +
                   'Action error: space is {}, action sent is {}\nOR\n' +
                   'Network error [socket doesnt exists or closed]: {}\n').\
                       format(self.action_space,
                              action,
                              not self.socket or self.socket.closed,)
            self.log.info(msg)
            raise AssertionError(msg)

        # Send action to backtrader engine, recieve response
        self.socket.send_pyobj(self.server_actions[action])
        self.server_response = self.socket.recv_pyobj()

        # Check if we really got that dict:
        try:
            assert type(self.server_response) == tuple and len(self.server_response) == 4

        except:
            msg = 'Environment response is: {}\nHint: Forgot to call reset()?'.format(self.server_response)
            raise AssertionError(msg)

        self.log.debug('Env.step() recieved response:\n{}\nAs type: {}'.
                       format(self.server_response, type(self.server_response)))

        return self.server_response

    def _close(self):
        """
        [kind of] Implementation of OpenAI Gym env.close method.
        Puts BTgym server in Control Mode:
        """
        _ = self._force_control_mode()
        # maybe TODO something

    def get_stat(self):
        """
        Returns last episode statistics.
        Note: when invoked, forces running episode to terminate.
        """
        if self._force_control_mode():
            self.socket.send_pyobj('_getstat')
            return self.socket.recv_pyobj()

        else:
            return self.server_response

