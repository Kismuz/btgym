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


import multiprocessing
import time
import datetime
import zmq

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding, closer

import numpy as np

import logging
logger = logging.getLogger(__name__)

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas


class BacktraderEnv(gym.Env, ):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 datafilename='./DAT_ASCII_EURUSD_M1_201702.csv',
                 net_address='tcp://127.0.0.1:5500',
                 max_episode_len=1000):
        # Server/network parameters:
        self.server = None
        self.network_address = net_address
        self.client_socket = None
        # RL env. related parameters:
        self.max_episode_len = max_episode_len
        self.observation_space = None
        self.action_space = spaces.Discrete(3)
        # Backtrader related parameters:
        self.cerebro_params = None
        # CSV parsing parameters:
        # These are specific to generic ASCII M1 bars FOREX data
        # from HistData.com
        self.datafeed_params = (
            ('dataname', datafilename),
            ('nullvalue', 0.0),
            ('dtformat', '%Y%m%d %H%M%S'),
            ('headers', True),
            ('separator', ';'),
            ('datetime', 0),
            ('high', 1),
            ('low', 2),
            ('open', 3),
            ('close', 4),
            ('volume', 5),
            ('openinterest', -1))
        # Run Backtrader server
        self._start_server()

    def _start_server(self):
        """
        Starts backtrader REQ/REP server as separate process
        and opens client comm. socket
        """
        # DBG
        print('Start(): starting server, max episode length:', self.max_episode_len)
        # Start server process
        self.server = multiprocessing.Process(target=bt_server_process,
                                              args=(self.datafeed_params,
                                                    self.max_episode_len,
                                                    self.cerebro_params,
                                                    self.network_address))
        self.server.daemon = False
        self.server.start()
        # Wait for server to startup
        time.sleep(5)
        # DBG
        print('Start(): server started with pid=', self.server.pid)
        # Set up client channel:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.network_address)
        # DBG
        print('Start(): client socket connected @ {}, pinging Backtrader:'.format(self.network_address))
        self.socket.send_pyobj('ping!')
        self.control_response = self.socket.recv_pyobj()
        print('Start(): received response: <{}>'.format(self.control_response))

    def _reset(self):
        """
        Implementation of OpenAI env.reset method.
        Rewinds backtrader server and starts new episode.
        Returns initial environment observations.
        """
        self.socket.send_pyobj('reset')
        self.control_response = self.socket.recv_pyobj()
        return self.control_response

    def _step(self, action):
        """
        Implementation of OpenAI env.step method.
        Relies on remote backtrader server for actual environment dynamics computing
        """
        # DBG
        print('Step(): sending action', action)
        # Send action to engine
        self.socket.send_pyobj(action)
        # Recieve response
        self.step_response = self.socket.recv_pyobj()
        # DBG
        print('Step(): recieved response {} as {}'.format(self.step_response, type(self.step_response)))
        return self.step_response

    def _compute_reward(self):
        """
        Computes RL reward based on backtrader current portfolio estimates.
        Actual reward function depends on how 'performance' is defined.
        """
        # TODO: not implemented
        self.reward = 0  # = TrickyRewardFunction[self.step_response]

        return self.reward

    def _close(self):
        """
        Shuts BT server down, destroys comm. chanell
        """
        self.socket.send_pyobj('stop')
        self.control_response = self.socket.recv_pyobj()
        # Ensure server exited:
        time.sleep(5)
        # Why take chances?
        self.server.terminate()
        self.server.join()
        self.socket.disconnect()
        self.context.destroy()
        return self.control_response


def bt_server_process(dataparams,
                      max_steps,
                      cerebro_params,
                      network_address):
    """
    Backtrader server.
    Control signals:
    IN:
    'reset' - rewinds backtrader engine and runs new episode;
    'stop' - server shut-down.
    OUT:
    info: <string message> - reports current server status.
    Within-episode signals:
    IN:
    <integer> - encoded action;
    'done' - stops current episode.
    OUT:
    response - <dict>: observation - observation of the current environment 
                                as [m,n] array of <fl32>, where:
                                n - num. of last datafeed values,
                                m - num. of data features;
                       reward - current portfolio statistics for environment reward signal computing;
                       done - episode termination flag;
                       info - auxiliary diagnostic information, if any.

    Parameters:
    dataparms - CSV file name and parsing parameters;
    max_steps - <int>, maximum episode length;
    network_address - <str>, network address to bind to;
    cerebro_params - <dict>: trading engine-specific parameters, excl. datafeed 
    """

    class FxCSVData(btfeeds.GenericCSVData):
        """
        Backtrader CSV datafeed class
        """
        params = dataparams

    class WorkHorseStrategy(bt.Strategy):
        """
        Defines actual environment inner dynamics.
        """
        params = (('socket', None),
                  ('max_steps', 0),
                  ('process', None),
                  ('iterations', 0),)

        def __init__(self):
            # Set parameters using args passed to bt_server_process
            self.p.process = multiprocessing.current_process()
            self.p.max_steps = max_steps
            print(info('process pid='), self.process.pid)

        def prenext(self):
            print('prenext() envoked.')

        def episode_stop(self):
            """
            Surprisingly, stops current episode 
            """
            self.env.runstop()

        def next(self):
            """
            Defines one step environment routine
            """
            is_done = False
            self.p.iterations += 1
            # Is it last step of the episode?
            if self.p.iterations >= self.p.max_steps:
                is_done = True
                # DBG
            print('Server: waiting for input')
            action_input = self.p.socket.recv_pyobj()
            # DBG
            print('Server: recieved {} as {}'.format(action_input, type(action_input)))
            # If client forces to finish episode:
            if action_input == 'done':
                action_input = None  # Some predefined action e.g. <close all positions>
                is_done = True

            # Do some mindfull computing...
            # is_done flag can also be rised here by
            # trading stimulator event, e.g. <OMG, we became too rich!>
            time.sleep(1)
            # STATE
            # REWARD
            # INFO
            # Compose and send response
            response = {'state': 'STATE', 'reward': 0, 'done': is_done, 'info': 'no information'}
            self.p.socket.send_pyobj(response)
            # DBG
            print('Server: response sent.')

            if is_done: self.episode_stop()

    def info(message):
        # Look who's talking
        return 'BT server: ' + message

        #### MAIN bt_server_process:

    # The best is yet to come:
    wonderful_results = []
    some_important_statisitc = []
    # Set up a comm. channel for server as zmq socket
    # to carry both service and data signal
    # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(network_address)

    # Main loop
    while True:  # TODO: use itertools here -> define global step
        # Server is in 'control' mode:
        service_input = socket.recv_pyobj()
        while service_input != 'reset':
            print(info('waiting for reset/stop signal'))
            socket.send_pyobj(info('waiting for reset/stop signal'))
            service_input = socket.recv_pyobj()
            # Check if we done with training:
            if service_input == 'stop':
                # Release comm channel, gather statistic and exit:
                # TODO: Gather overall statistics
                # all the server shutdown logic is here
                some_important_statisitc = True
                # To leave gracefully:
                print(info('exiting.'))
                socket.send_pyobj(info('exiting.'))
                socket.close()
                context.destroy()
                return wonderful_results, some_important_statisitc
                # And where do you think you actually return it, hah?  <-- TODO: dump stats to file or something

        # Got 'reset'? --> start new episode:
        print(info('starting new episode'))
        socket.send_pyobj(info('starting new episode'))  # just to pair control message

        # <some fancy code to define random data entry point for upcoming episode>
        # <i.e. 'fromdate'  and 'todate' parameters>

        datafeed = FxCSVData(fromdate=datetime.datetime(2017, 2, 1),
                             todate=datetime.datetime(2017, 2, 2), )
        # Compose bt.cerebro class for the episode:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(WorkHorseStrategy,
                            socket=socket,
                            iterations=0)
        cerebro.adddata(datafeed)
        wonderful_results = cerebro.run()

    # Just in case -- we actually cannot get there except by some ignored exception
    return wonderful_results, some_important_statisitc

