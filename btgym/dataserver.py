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
            dataset:            BTgymDataset instance;
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
        self.dataset.read_csv()

        # Describe dataset:
        try:
            assert not self.dataset.data_stat.empty
            pass

        except:
            _ = self.dataset.describe()

        self.dataset_stat = self.dataset.data_stat

        local_step = 0
        # Main loop:
        while True:
            # Get random episode dataset:
            episode_dataset = self.dataset.sample_random()

            # Compose response:
            data_dict = dict(
                datafeed=episode_dataset.to_btfeed(),
                episode_stat=episode_dataset.describe(),
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
                    message = 'DataServer is exiting.'
                    self.log.info(message)
                    socket.send_pyobj(message)
                    socket.close()
                    context.destroy()
                    return None

                # Send episode datafeed:
                elif service_input['ctrl'] == '_get_data':
                    message = 'Sending episode #{} data.'.format(local_step)
                    self.log.debug(message)
                    socket.send_pyobj(data_dict)
                    local_step += 1

                    # Send dataset statisitc:
                elif service_input['ctrl'] == '_get_info':
                    message = 'Sending info.'.format(local_step)
                    self.log.debug(message)
                    # Compose response:
                    info_dict = dict(
                        dataset_stat=self.dataset_stat,
                        dataset_columns=list(self.dataset.names),
                        pid=self.process.pid,
                    )
                    socket.send_pyobj(info_dict)

                else:  # ignore any other input
                    # NOTE: response dictionary must include 'ctrl' key
                    message = {'ctrl': 'send control keys:  <_get_data>, <_get_info>, <_stop>.'}
                    self.log.debug('DataServer sent: ' + str(message))
                    socket.send_pyobj(message)  # pairs any other input

            else:
                message = 'No <ctrl> key received:{}\n'.format(msg)
                self.log.debug(message)
                socket.send_pyobj(message) # pairs input
