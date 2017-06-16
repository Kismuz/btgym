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
    response = None

    def __init__(self):
        """
        Inherit logger and ZMQ socket from parent:
        """
        self.log = self.strategy.env._log
        self.socket = self.strategy.env._socket

    def prenext(self):
        pass

    def stop(self):
        pass

    def next(self):
        """
        Actual env.step() communication and episode termination is here.
        """

        # Gather response:
        self.strategy.get_state()
        self.strategy.get_reward()
        self.strategy._get_done()
        self.strategy.get_done()
        self.strategy.get_info()

        # Halt and wait to receive action from outer world:
        self.strategy.action = self.socket.recv_pyobj()
        self.log.debug('COMM recieved: {}'.format(self.strategy.action))
        self.response = {'state': self.strategy.state,
                         'reward': self.strategy.reward,
                         'done': self.strategy.is_done,
                         'info': self.strategy.info}
        # Send response:
        self.socket.send_pyobj(self.response)
        #self.log.debug('COMM sent: {}//{}'.format(self.response['done'], self.response['info']))

        # If done, initiate fallback to Control Mode:
        if self.strategy.is_done:
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

    Control signals:
    IN:
    '_reset' - rewinds backtrader engine and runs new episode;
    '_getstat' - retrieve episode results and statistics;
    '_stop' - server shut-down.
    OUT:
    <string message> - reports current server status;
    <statisic dict> - last run episode statisics.  NotImplemented.

    Within-episode signals:
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
                       info - auxiliary information.

    Parameters:
    datafeed  - class BTgymData instance;
    cerebro - subclass bt.Cerebro;
    network_address - <str>, network address to bind to;
    verbose - verbosity mode: 0 - silent, 1 - info level, 2 - debugging level
    """

    def __init__(self,
                 datafeed=None,
                 cerebro=None,
                 network_address=None,
                 verbose=False):
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
        if not datafeed:
            raise AssertionError('Server has not recieved any datafeed. Nothing to run!')
        else:
            self.datafeed = datafeed

        # Net:
        if not network_address:
            raise AssertionError('Server has not recieved network address to bind to!')
        else:
            self.network_address = network_address

        self.verbose = verbose

    def run(self):
        """
        Server process runtime body. This method is invoked by env._start_server().
        """
        # Verbosity control:
        if self.verbose:
            if self.verbose == 2:
                logging.getLogger().setLevel(logging.DEBUG)
            else:
                logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        log = logging.getLogger('BTgym_server')

        self.process = multiprocessing.current_process()
        log.info('Server process PID: {}'.format(self.process.pid))

        # Runtime Housekeeping:
        episode_result = dict()

        # Set up a comm. channel for server as ZMQ socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        # Actually load data to BTgymData instance:
        self.datafeed.read_csv()

        # Add logging:
        self.datafeed.log = log

        # Server 'Control Mode' loop:
        for episode_number in itertools.count(1):

            # Stuck here until '_reset' or '_stop':
            while True:
                service_input = socket.recv_pyobj()
                log.debug('Server Control mode: recieved <{}>'.format(service_input))

                # Check if it's time to exit:
                if service_input == '_stop':
                    # Server shutdown logic:
                    # send last run statistic, release comm channel and exit:
                    message = 'Server is exiting.'
                    log.info(message)
                    socket.send_pyobj(message)
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

                else:  # ignore any other input
                    # NOTE: response string must include 'CONTROL' exact substring
                    # for env.reset(), env.get_stat(), env.close() correct operation.
                    message = 'CONTROL mode, send <_reset>, <_getstat> or <_stop>.'
                    log.debug('Server sent: ' + message)
                    socket.send_pyobj(message)  # pairs any other input

            # Got '_reset' signal, prepare Cerebro subclass and run episode:
            cerebro = copy.deepcopy(self.cerebro)
            cerebro._socket = socket
            cerebro._log = log

            # Add DrawdDown observer if not already:
            dd_added = False
            for observer in cerebro.observers:

                if bt.observers.DrawDown in observer:
                    dd_added = True

            if not dd_added:
                cerebro.addobserver(bt.observers.DrawDown)

            # Add communication utility:
            cerebro.addanalyzer(_BTgymAnalyzer,
                                _name='_env_analyzer',)

            # Add random episode data:
            cerebro.adddata(self.datafeed.sample_random().to_btfeed())

            # Finally:
            episode = cerebro.run(stdstats=True, preload=True)[0]
            log.info('Episode finished.')

            # Recover that bloody analytics:
            analyzers_list = episode.analyzers.getnames()
            analyzers_list.remove('_env_analyzer')

            episode_result['episode'] = episode_number

            for name in analyzers_list:
                episode_result[name] = episode.analyzers.getbyname(name).get_analysis()


        # Just in case -- we actually shouldnt get there except by some error:
        return None
