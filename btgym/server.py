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
import time
import datetime
import random
import itertools
import zmq
import copy

import backtrader as bt
import backtrader.feeds as btfeeds

###################### BT Server in-episode communocation method ##############


class _EpisodeComm(bt.Analyzer):
    """
    Adding this [misused] analyzer to cerebro instance enables strategy REQ/REP communication while in episode mode.
    No, as part of core server operational logic, it should not be explicitly called/edited.
    Yes, it actually analyzes nothing.
    """
    log = None
    socket = None
    response = None

    def __init__(self):
        """
        Inherit .log and ZMQ socket from parent.
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

############################## Episodic Datafeed Class #########################


class TestDataLen(bt.Strategy):
    """
    Service strategy, only used by <EpisodicDataFeed>.test_data_period() method.
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
    # TODO: make it faster

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

            # TODO: Can loop forever, if something is wrong with data, etc.
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
                 network_address,
                 cerebro=None,
                 verbose=False):
        """
        Configures BT server instance.
        """
        super(BTserver, self).__init__()
        self.dataparams = dataparams
        self.network_address = network_address
        self.verbose = verbose

        # Cerebro class to execute:
        if not cerebro:
            raise AssertionError('Server has not recieved any bt.cerebro() class. Nothing to run!')
        else:
            self.cerebro = cerebro

        # Configure datafeed subclass:
        class CSVData(btfeeds.GenericCSVData):
            """Backtrader CSV datafeed class"""
            params = self.dataparams
            params['numrecords'] = 0
        self.data = EpisodicDataFeed(CSVData)

    def run(self):
        """
        Server process execution body. This method is evoked by env._start_server().
        """
        # Verbosity control:
        if self.verbose:
            if self.verbose == 2:
                logging.getLogger().setLevel(logging.DEBUG)
            else:
                logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        log = logging.getLogger('BT_server')

        self.process = multiprocessing.current_process()
        log.info('Server process PID: {}'.format(self.process.pid))

        # Housekeeping:
        cerebro_result = 'No runs has been made.'

        # Set up a comm. channel for server as ZMQ socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        # Lookup datafeed:
        self.data.measure()

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

            # Add communication utility:
            cerebro.addanalyzer(_EpisodeComm,
                                _name='communicator',)

            # Add random episode data:
            cerebro.adddata(self.data.sample_episode(cerebro.min_episode_len,
                                                     cerebro.max_episode_days))

            # Finally:
            episode = cerebro.run(stdstats=True)[0]
            log.info('Episode finished.')
            # TODO: finally make that stat passing over!
            # Get statistics:
            episode_result = dict(episode = episode_number,)
                                  #stats = episode.stats,
                                  #analyzers = episode.analyzers,
            log.debug('ANALYZERS: {}'.format(len(episode.analyzers)))
            log.debug('DATAFEEDS: {}'.format(len(episode.datas)))

        # Just in case -- we actually shouldnt get there except by some error:
        return None
