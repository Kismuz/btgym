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
import copy
import zmq
import datetime

from .datafeed import DataSampleConfig


class BTgymDataFeedServer(multiprocessing.Process):
    """
    Data provider server class.
    Enables efficient data sampling for asynchronous multiply BTgym environments execution.
    Manages global back-testing time and broadcast messages.
    """
    process = None
    dataset_stat = None

    def __init__(self, dataset=None, network_address=None, log_level=None, task=0):
        """
        Configures data server instance.

        Args:
            dataset:            data domain instance;
            network_address:    ...to bind to.
            log_level:          int, logbook.level
            task:               id
        """
        super(BTgymDataFeedServer, self).__init__()

        self.log_level = log_level
        self.task = task
        self.log = None
        self.local_step = 0
        self.dataset = dataset
        self.network_address = network_address
        self.default_sample_config = copy.deepcopy(DataSampleConfig)
        self.broadcast_message = None

        self.debug_pre_sample_fails = 0
        self.debug_pre_sample_attempts = 0

        # self.global_timestamp = 0

    def get_data(self, sample_config=None):
        """
        Get Trial sample according to parameters received.
        If no parameters being passed - makes sample with default parameters.

        Args:
            sample_config:   sampling parameters configuration dictionary

        Returns:
            sample:     if `sample_params` arg has been passed and dataset is ready
            None:       otherwise
        """
        if self.dataset.is_ready:
            if sample_config is not None:
                # We do not allow configuration timestamps which point earlier than current global_timestamp;
                # if config timestamp points later - it is ok because global time will be shifted accordingly after
                # [traget test] sample will get into work.
                if sample_config['timestamp'] is None:
                    sample_config['timestamp'] = 0

                # If config timestamp is outdated - refresh with latest:
                if sample_config['timestamp'] < self.dataset.global_timestamp:
                    sample_config['timestamp'] = copy.deepcopy(self.dataset.global_timestamp)

                self.log.debug('Sampling with params: {}'.format(sample_config))
                sample = self.dataset.sample(**sample_config)

            else:
                self.default_sample_config['timestamp'] = copy.deepcopy(self.dataset.global_timestamp)
                self.log.debug('Sampling with default params: {}'.format(self.default_sample_config))
                sample = self.dataset.sample(**self.default_sample_config)

            self.local_step += 1

        else:
            # Dataset not ready, make dummy:
            sample = None

        return sample

    def run(self):
        """
        Server process runtime body.
        """
        # Logging:
        from logbook import Logger, StreamHandler, WARNING
        import sys
        StreamHandler(sys.stdout).push_application()
        if self.log_level is None:
            self.log_level = WARNING
        self.log = Logger('BTgymDataServer_{}'.format(self.task), level=self.log_level)

        self.process = multiprocessing.current_process()
        self.log.info('PID: {}'.format(self.process.pid))

        # Set up a comm. channel for server as ZMQ socket:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        try:
            assert not self.dataset.data.empty

        except (AssertionError, AttributeError) as e:
            self.log.error("no data defined.")
            raise e     # fail the app

        # Describe dataset:
        self.dataset_stat = self.dataset.describe()

        # Main loop:
        while True:
            # Stick here until receive any request:
            service_input = socket.recv_pyobj()
            self.log.debug('Received <{}>'.format(service_input))

            if 'ctrl' in service_input:
                # It's time to exit:
                if service_input['ctrl'] == '_stop':
                    # Server shutdown logic:
                    # send last run statistic, release comm channel and exit:
                    message = {'ctrl': 'Exiting.'}
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
                    # self.global_timestamp = self.dataset.global_timestamp
                    self.log.notice(
                        'Initial global_time set to: {} / stamp: {}'.
                        format(
                            datetime.datetime.fromtimestamp(self.dataset.global_timestamp),
                            self.dataset.global_timestamp
                        )
                    )
                    message = {'ctrl': 'Reset with kwargs: {}'.format(kwargs)}
                    self.log.debug('Data_is_ready: {}'.format(self.dataset.is_ready))
                    socket.send_pyobj(message)
                    self.local_step = 0

                # Send dataset sample:
                elif service_input['ctrl'] == '_get_data':
                    if self.dataset.is_ready:
                        sample = self.get_data(sample_config=service_input['kwargs'])
                        message = 'Sending sample_#{}.'.format(self.local_step)
                        self.log.debug(message)
                        socket.send_pyobj(
                            {
                                'sample': sample,
                                'stat': self.dataset_stat,
                                'origin': 'data_server',
                                'timestamp': self.dataset.global_timestamp,
                            }
                        )

                    else:
                        message = {'ctrl': 'Dataset not ready, waiting for control key <_reset_data>'}
                        self.log.debug('Sent: ' + str(message))
                        socket.send_pyobj(message)  # pairs any other input

                # Send dataset statisitc:
                elif service_input['ctrl'] == '_get_info':
                    message = 'Sending info for #{}.'.format(self.local_step)
                    self.log.debug(message)
                    # Compose response:
                    info_dict = dict(
                        dataset_stat=self.dataset_stat,
                        dataset_columns=list(self.dataset.names),
                        pid=self.process.pid,
                        dataset_is_ready=self.dataset.is_ready,
                        data_names=self.dataset.data_names
                    )
                    socket.send_pyobj(info_dict)

                # Set global time:
                elif service_input['ctrl'] == '_set_broadcast_message':
                    if self.dataset.global_timestamp != 0 and self.dataset.global_timestamp > service_input['timestamp']:
                        message = 'Moving back in time not supported! ' +\
                                  'Current global_time: {}, '.\
                                      format(datetime.datetime.fromtimestamp(self.dataset.global_timestamp)) +\
                                  'attempt to set: {}; global_time and broadcast message not set.'.\
                                      format(datetime.datetime.fromtimestamp(service_input['timestamp'])) +\
                                  'Hint: check sampling logic consistency.'

                        self.log.info(message)

                    else:
                        self.dataset.global_timestamp = service_input['timestamp']
                        self.broadcast_message = service_input['broadcast_message']
                        message = 'global_time set to: {} / stamp: {}'.\
                            format(
                                datetime.datetime.fromtimestamp(self.dataset.global_timestamp),
                                self.dataset.global_timestamp
                            )
                    socket.send_pyobj(message)
                    self.log.debug(message)

                elif service_input['ctrl'] == '_get_global_time':
                    # Tell time:
                    message = {'timestamp': self.dataset.global_timestamp}
                    socket.send_pyobj(message)

                elif service_input['ctrl'] == '_get_broadcast_message':
                    # Tell:
                    message = {
                        'timestamp': self.dataset.global_timestamp,
                        'broadcast_message': self.broadcast_message,
                    }
                    socket.send_pyobj(message)

                else:  # ignore any other input
                    # NOTE: response dictionary must include 'ctrl' key
                    message = {
                        'ctrl':
                            'waiting for control keys:  <_reset_data>, <_get_data>, ' +
                            '<_get_info>, <_stop>, <_get_global_time>, <_get_broadcast_message>'
                    }
                    self.log.debug('Sent: ' + str(message))
                    socket.send_pyobj(message)  # pairs any other input

            else:
                message = {'ctrl': 'No <ctrl> key received, got:\n{}'.format(service_input)}
                self.log.debug(str(message))
                socket.send_pyobj(message) # pairs input
