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

import logging
#logging.basicConfig(format='%(name)s: %(message)s')
import multiprocessing

import itertools
import zmq
import copy

import time
from datetime import timedelta

import backtrader as bt

###################### BT Server in-episode communocation method ##############


class _BTgymAnalyzer(bt.Analyzer):
    """
    This [kind of] misused analyzer handles strategy/environment communication logic
    while in episode mode.
    As part of core server operational logic, it should not be explicitly called/edited.
    Yes, it actually analyzes nothing.
    """
    log = None
    socket = None

    def __init__(self):
        """
        Inherit logger and ZMQ socket from parent:
        """
        self.log = self.strategy.env._log
        self.socket = self.strategy.env._socket
        self.message = None
        self.info_list = []

    def prenext(self):
        pass

    def stop(self):
        pass

    def next(self):
        """
        Actual env.step() communication and episode termination is here.
        """
        # We'll do it every step:
        # If it's time to leave:
        is_done = self.strategy._get_done()
        # Collect step info:
        self.info_list.append(self.strategy.get_info())
        # Put agent on hold:
        self.strategy.action = 'hold'

        # Only if it's time to communicate or episode has come to end:
        if self.strategy.iteration % self.strategy.p.skip_frame == 0 or is_done:

            # Gather response:
            state = self.strategy.get_state()
            reward = self.strategy.get_reward()

            # Halt and wait to receive message from outer world:
            self.message= self.socket.recv_pyobj()
            msg = 'COMM recieved: {}'.format(self.message)
            self.log.debug(msg)

            # Paraniod check:
            try:
                self.strategy.action = self.message['action']

            except:
                msg = 'No <action> key recieved:\n' + msg
                raise AssertionError(msg)

            # Send response as <o, r, d, i> tuple (Gym convention):
            self.socket.send_pyobj((state, reward, is_done, self.info_list))

            # Reset info:
            self.info_list = []

        # If done, initiate fallback to Control Mode:
        if is_done:
            self.log.debug('RunStop() invoked with {}'.format(self.strategy.broker_message))
            self.strategy.close()
            self.strategy.env.runstop()

        # Strategy housekeeping:
        self.strategy.iteration += 1
        self.strategy.broker_message = '-'

##############################  BTgym Server Main  ##############################

class BTgymServer(multiprocessing.Process):
    """
    Backtrader server class.

    Expects to receive dictionary, containing at least 'action' field.

    Control mode IN:
    dict(action=<control action, type=str>,), where
    control action is:
    '_reset' - rewinds backtrader engine and runs new episode;
    '_getstat' - retrieve episode results and statistics;
    '_stop' - server shut-down.

    OUT:
    <string message> - reports current server status;
    <statisic dict> - last run episode statisics.  NotImplemented.

    Within-episode signals:
    Episode mode IN:
    dict(action=<agent_action, type=str>,), where
    agent_action is:
    {'buy', 'sell', 'hold', 'close', '_done'} - agent or service actions; '_done' - stops current episode;

    OUT:
    response  <tuple>: observation, <array> - observation of the current environment state,
                                             could be any tensor; default is:
                                             [4,m] array of <fl32>, where:
                                             m - num. of last datafeed values,
                                             4 - num. of data features (Lines);
                       reward, <any> - current portfolio statistics for environment reward estimation;
                       done, <bool> - episode termination flag;
                       info, <list> - auxiliary information.

    Parameters:
    datafeed  - class BTgymDataset instance;
    cerebro -  bt.Cerebro engine subclass;
    network_address - <str>, network address to bind to;
    verbose - verbosity mode: 0 - silent, 1 - info level, 2 - debugging level
    """

    def __init__(self,
                 dataset=None,
                 cerebro=None,
                 network_address=None,
                 log=None):
        """
        Configures BT server instance.
        """

        super(BTgymServer, self).__init__()

        # Paranoid checks:
        # Cerebro class to execute:
        if not cerebro:
            raise AssertionError('Server has not recieved any bt.cerebro() class. Nothing to run!')
        else:
            self.cerebro = cerebro

        # Datafeed instance to load from:
        if not dataset:
            raise AssertionError('Server has not recieved any datafeed. Nothing to run!')
        else:
            self.dataset = dataset

        # Net:
        if not network_address:
            raise AssertionError('Server has not recieved network address to bind to!')
        else:
            self.network_address = network_address

        # To log or not to log:
        if not log:
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

        else:
            self.log = log

    def run(self):
        """
        Server process runtime body. This method is invoked by env._start_server().
        """
        self.process = multiprocessing.current_process()
        self.log.info('Server PID: {}'.format(self.process.pid))

        # Runtime Housekeeping:
        episode_result = dict()

        # Set up a comm. channel for server as ZMQ socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        # Actually load data to BTgymDataset instance:
        self.dataset.read_csv()

        # Describe dataset if not already and pass it to strategy params:
        try:
            assert not self.dataset.data_stat.empty
            pass

        except:
            _ = self.dataset.describe()

        self.cerebro.strats[0][0][2]['dataset_stat'] = self.dataset.data_stat

        # Server 'Control Mode' loop:
        for episode_number in itertools.count(1):

            # Stuck here until '_reset' or '_stop':
            while True:

                service_input = socket.recv_pyobj()
                msg = 'Server Control mode: recieved <{}>'.format(service_input)
                self.log.debug(msg)

                try:
                    assert 'action' in service_input

                except:
                    msg = 'No <action> key recieved:\n' + msg
                    raise AssertionError(msg)

                # Check if it's time to exit:
                if service_input['action'] == '_stop':
                    # Server shutdown logic:
                    # send last run statistic, release comm channel and exit:
                    message = 'Server is exiting.'
                    self.log.info(message)
                    socket.send_pyobj(message)
                    socket.close()
                    context.destroy()
                    return None

                elif service_input['action'] == '_reset':
                    message = 'Starting episode.'
                    self.log.info(message)
                    socket.send_pyobj(message)  # pairs '_reset'
                    break

                elif service_input['action'] == '_getstat':
                    socket.send_pyobj(episode_result)
                    self.log.info('Episode statistic sent.')

                else:  # ignore any other input
                    # NOTE: response string must include 'CONTROL_MODE' exact substring
                    # for env.reset(), env.get_stat(), env.close() correct operation.
                    message = 'CONTROL_MODE, send <_reset>, <_getstat> or <_stop>.'
                    self.log.debug('Server sent: ' + message)
                    socket.send_pyobj(message)  # pairs any other input

            # Got '_reset' signal -> prepare Cerebro subclass and run episode:
            cerebro = copy.deepcopy(self.cerebro)
            cerebro._socket = socket
            cerebro._log = self.log

            # Add DrawDown observer if not already:
            dd_added = False
            for observer in cerebro.observers:

                if bt.observers.DrawDown in observer:
                    dd_added = True

            if not dd_added:
                cerebro.addobserver(bt.observers.DrawDown)

            # Add communication utility:
            cerebro.addanalyzer(_BTgymAnalyzer,
                                _name='_env_analyzer',)

            # Get random episode dataset:
            episode_dataset = self.dataset.sample_random()

            # Get episode data statistic and pass it to strategy params:
            cerebro.strats[0][0][2]['episode_stat'] = episode_dataset.describe()

            # Add data to engine:
            cerebro.adddata(episode_dataset.to_btfeed())

            # Finally:
            start_time = time.time()
            episode = cerebro.run(stdstats=True, preload=False)[0]

            elapsed_time = timedelta(seconds=time.time() - start_time)
            self.log.info('Episode elapsed time: {}.'.format(elapsed_time))

            # Recover that bloody analytics:
            analyzers_list = episode.analyzers.getnames()
            analyzers_list.remove('_env_analyzer')

            episode_result['episode'] = episode_number
            episode_result['runtime'] = elapsed_time

            for name in analyzers_list:
                episode_result[name] = episode.analyzers.getbyname(name).get_analysis()

        # Just in case -- we actually shouldn't get there except by some error:
        return None
