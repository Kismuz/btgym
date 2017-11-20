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

#import itertools
import zmq

#import time
#from datetime import timedelta


class BTgymDataFeedServer(multiprocessing.Process):
    """
    Data provider server class.
    Enables efficient data sampling for asynchronous multiply BTgym environments execution.
    """
    process = None
    dataset_stat = None

    def __init__(self, dataset=None, network_address=None, log=None):
        """
        Configures data server instance.

        Args:
            dataset:            BTgymDataset or othe rdata provider class instance;
            network_address:    ...to bind to.
            log:                parent logger.
        """
        super(BTgymDataFeedServer, self).__init__()

        # To log or not to log:
        if log is None:
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

        else:
            self.log = log

        self.dataset = dataset
        self.network_address = network_address

    def run(self):
        """
        Server process runtime body.
        """
        self.process = multiprocessing.current_process()
        self.log.info('DataServer PID: {}'.format(self.process.pid))

        # Set up a comm. channel for server as ZMQ socket:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        # Actually load data to BTgymDataset instance:
        try:
            assert not self.dataset.data.empty

        except (AssertionError, AttributeError) as e:
            self.dataset.read_csv()

        # Describe dataset:
        self.dataset_stat = self.dataset.describe()

        local_step = 0
        fresh_sample = False

        # Main loop:
        while True:
            self.log.debug('If_sample: data_ready: {}, fresh_sample: {}'.format(self.dataset.is_ready, fresh_sample))
            if not fresh_sample:
                if self.dataset.is_ready:
                    # Get random episode dataset:
                    episode = self.dataset.sample()
                    # Compose response:
                    data_dict = dict(
                        metadata=episode.metadata,
                        datafeed=episode.to_btfeed(),
                        episode_stat=episode.describe(),
                        dataset_stat=self.dataset_stat,
                        local_step=local_step,
                    )
                    fresh_sample = True
                    self.log.debug('Got fresh: episode #{} metadata:\n{}'.format(local_step, episode.metadata))

                else:
                    # Dataset not ready, make dummy:
                    data_dict = dict(
                        metadata=None,
                        datafeed=None,
                        episode_stat=None,
                        dataset_stat=self.dataset_stat,
                        local_step=local_step,
                    )

            # Stick here with episode data in hand until get request:
            service_input = socket.recv_pyobj()
            msg = 'DataServer received <{}>'.format(service_input)
            self.log.debug(msg)

            if 'ctrl' in service_input:
                # It's time to exit:
                if service_input['ctrl'] == '_stop':
                    # Server shutdown logic:
                    # send last run statistic, release comm channel and exit:
                    message = {'ctrl': 'DataServer is exiting.'}
                    self.log.info(str(message))
                    socket.send_pyobj(message)
                    socket.close()
                    context.destroy()
                    return None

                # Reset datafeed:
                elif service_input['ctrl'] == '_reset_data':
                    try:
                        kwargs = service_input['kwargs']

                    except KeyError:
                        kwargs = {}

                    self.dataset.reset(**kwargs)
                    message = {'ctrl': 'Dataset has been reset with kwargs: {}'.format(kwargs)}
                    self.log.debug('DataServer sent: ' + str(message))
                    self.log.debug('[_reset_data]: data_is_ready: {}'.format(self.dataset.is_ready))
                    socket.send_pyobj(message)
                    fresh_sample = False

                # Send episode datafeed:
                elif service_input['ctrl'] == '_get_data':
                    if self.dataset.is_ready:
                        message = 'Sending episode #{} data {}.'.format(local_step, data_dict)
                        self.log.debug(message)
                        socket.send_pyobj(data_dict)
                        local_step += 1

                    else:
                        message = {'ctrl': 'Dataset not ready, waiting for control key <_reset_data>'}
                        self.log.debug('DataServer sent: ' + str(message))
                        socket.send_pyobj(message)  # pairs any other input
                    # Mark current sample as used anyway:
                    fresh_sample = False

                # Send dataset statisitc:
                elif service_input['ctrl'] == '_get_info':
                    message = 'Sending info for #{}.'.format(local_step)
                    self.log.debug(message)
                    # Compose response:
                    info_dict = dict(
                        dataset_stat=self.dataset_stat,
                        dataset_columns=list(self.dataset.names),
                        pid=self.process.pid,
                        dataset_is_ready=self.dataset.is_ready
                    )
                    socket.send_pyobj(info_dict)

                else:  # ignore any other input
                    # NOTE: response dictionary must include 'ctrl' key
                    message = {'ctrl': 'waiting for control keys:  <_reset_data>, <_get_data>, <_get_info>, <_stop>.'}
                    self.log.debug('DataServer sent: ' + str(message))
                    socket.send_pyobj(message)  # pairs any other input

            else:
                message = {'ctrl': 'No <ctrl> key received, got:\n{}'.format(msg)}
                self.log.debug(str(message))
                socket.send_pyobj(message) # pairs input
