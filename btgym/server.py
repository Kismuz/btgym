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
import gc
import psutil

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
        # Inherit logger and ZMQ socket from parent:
        self.log = self.strategy.env._log
        self.socket = self.strategy.env._socket
        self.render = self.strategy.env._render
        self.message = None
        self.step_to_render = None # Due to reset(), this will get populated before first render() call.

        # At the end of the episode - render everything but episode:
        self.render_at_stop = self.render.render_modes.copy()
        try:
            self.render_at_stop.remove('episode')

        except:
            pass

        self.info_list = []

    def prenext(self):
        pass

    def stop(self):
        pass

    def early_stop(self):
        """
        Stop, take picture and get out.
        """
        self.log.debug('RunStop() invoked with {}'.format(self.strategy.broker_message))

        # Do final renderings, it will be kept by renderer class, not sending anywhere:
        self.render.render(self.render_at_stop, step_to_render=self.step_to_render, send_img=False)

        self.strategy.close()
        self.strategy.env.runstop()

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
            raw_state = self.strategy._get_raw_state()
            state = self.strategy.get_state()
            # DUMMY:

            reward = self.strategy.get_reward()

            # Halt and wait to receive message from outer world:
            self.message = self.socket.recv_pyobj()
            msg = 'COMM recieved: {}'.format(self.message)
            self.log.debug(msg)

            # Control actions loop, ignoring 'action' key:
            while 'ctrl' in self.message:
                # Rendering requested:
                if self.message['ctrl'] == '_render':
                    self.socket.send_pyobj(
                        self.render.render(
                            self.message['mode'],
                            step_to_render=self.step_to_render,
                        )
                    )
                # Episode termination requested:
                if self.message['ctrl'] == '_done':
                    is_done = True  # redundant
                    self.socket.send_pyobj('_DONE SIGNAL RECEIVED')
                    self.early_stop()
                    return None

                # Halt again:
                self.message = self.socket.recv_pyobj()
                msg = 'COMM recieved: {}'.format(self.message)
                self.log.debug(msg)

            # Store agent action:
            if 'action' in self.message: # now it should!
                self.strategy.action = self.message['action']

            else:
                msg = 'No <action> key recieved:\n' + msg
                raise AssertionError(msg)

            # Send response as <o, r, d, i> tuple (Gym convention),
            # opt to send entire info_list or just latest part:
            info = [self.info_list[-1]]
            self.socket.send_pyobj((state, reward, is_done, info))

            # Back up step information for rendering.
            # It pays when using skip-frames: will'll get future state otherwise.

            self.step_to_render = ({'human':raw_state}, state, reward, is_done, self.info_list)

            # Reset info:
            self.info_list = []

        # If done, initiate fallback to Control Mode:
        if is_done:
            self.early_stop()

        # Strategy housekeeping:
        self.strategy.iteration += 1
        self.strategy.broker_message = '-'

    ##############################  BTgym Server Main  ##############################


class BTgymServer(multiprocessing.Process):
    """Backtrader server class.

    Expects to receive dictionary, containing at least 'action' field.

    Control mode IN::
        dict(action=<control action, type=str>,), where
        control action is:
        '_reset' - rewinds backtrader engine and runs new episode;
        '_getstat' - retrieve episode results and statistics;
        '_stop' - server shut-down.

    Control mode OUT::
        <string message> - reports current server status;
        <statisic dict> - last run episode statisics.  NotImplemented.

        Within-episode signals:
        Episode mode IN:
        dict(action=<agent_action, type=str>,), where
        agent_action is:
        {'buy', 'sell', 'hold', 'close', '_done'} - agent or service actions; '_done' - stops current episode;

    Episode mode OUT::
        response  <tuple>: observation, <array> - observation of the current environment state,
                                                 could be any tensor; default is:
                                                 [4,m] array of <fl32>, where:
                                                 m - num. of last datafeed values,
                                                 4 - num. of data features (Lines);
                           reward, <any> - current portfolio statistics for environment reward estimation;
                           done, <bool> - episode termination flag;
                           info, <list> - auxiliary information.
    """
    data_server_response = None

    def __init__(self,
                 cerebro=None,
                 render=None,
                 network_address=None,
                 data_network_address=None,
                 connect_timeout=60,
                 log=None):
        """

        Args:
            cerebro:                backtrader.cerebro engine class.
            render:                 render class
            network_address:        environmnet communication, str
            data_network_address:   data communication, str
            connect_timeout:        seconds, int
            log:                    none
        """

        super(BTgymServer, self).__init__()

        # To log or not to log:
        if not log:
            self.log = logging.getLogger('dummy')
            self.log.addHandler(logging.NullHandler())

        else:
            self.log = log

        self.cerebro = cerebro
        self.network_address = network_address
        self.render = render
        self.data_network_address = data_network_address
        self.connect_timeout = connect_timeout # server connection timeout in seconds.
        self.connect_timeout_step = 0.01

    def _comm_with_timeout(self, socket, message, timeout, connect_timeout_step=0.01,):
        """
        Exchanges messages via socket with timeout.

        Returns:
            dictionary:
                status: communication result;
                message: received message, if any.
        """
        response=dict(
            status='ok',
            message=None,
        )
        try:
            socket.send_pyobj(message)
        except:
            response['status'] = 'send_failed'
            return response

        for i in itertools.count():
            try:
                response['message'] = socket.recv_pyobj(flags=zmq.NOBLOCK)
                response['time'] = i * connect_timeout_step
                break

            except:
                time.sleep(connect_timeout_step)
                if i >= timeout / connect_timeout_step:
                    response['status'] = 'receive_failed'
                    return response

        return response

    def run(self):
        """
        Server process runtime body. This method is invoked by env._start_server().
        """
        self.process = multiprocessing.current_process()
        self.log.info('BTgymServer PID: {}'.format(self.process.pid))

        # Runtime Housekeeping:
        cerebro = None
        episode_result = dict()

        # Set up a comm. channel for server as ZMQ socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.network_address)

        data_context = zmq.Context()
        data_socket = data_context.socket(zmq.REQ)
        data_socket.connect(self.data_network_address)

        # Check connection:
        self.log.debug('BtgymServer: pinging data_server at: {} ...'.format(self.data_network_address))

        data_server_response = self._comm_with_timeout(
            socket=data_socket,
            message={'ctrl': 'ping!'},
            timeout=self.connect_timeout,
        )
        if data_server_response['status'] in 'ok':
            self.log.debug('BTgymServer: Data_server seems ready with response: <{}>'.
                          format(data_server_response['message']))

        else:
            msg = 'BtgymServer: data_server unreachable with status: <{}>.'.\
                format(data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)


        # Init renderer:
        self.render.initialize_pyplot()

        # Server 'Control Mode' loop:
        for episode_number in itertools.count(0):
            while True:
                # Stuck here until '_reset' or '_stop':
                service_input = socket.recv_pyobj()
                msg = 'Server Control mode: received <{}>'.format(service_input)
                self.log.debug(msg)

                if 'ctrl' in service_input:
                    # It's time to exit:
                    if service_input['ctrl'] == '_stop':
                        # Server shutdown logic:
                        # send last run statistic, release comm channel and exit:
                        message = 'Server is exiting.'
                        self.log.info(message)
                        socket.send_pyobj(message)
                        socket.close()
                        context.destroy()
                        return None

                    # Start episode:
                    elif service_input['ctrl'] == '_reset':
                        message = 'Starting episode.'
                        self.log.debug(message)
                        socket.send_pyobj(message)  # pairs '_reset'
                        break

                    # Retrieve statistic:
                    elif service_input['ctrl'] == '_getstat':
                        socket.send_pyobj(episode_result)
                        self.log.debug('Episode statistic sent.')

                    # Send episode rendering:
                    elif service_input['ctrl'] == '_render' and 'mode' in service_input.keys():
                        # Just send what we got:
                        socket.send_pyobj(self.render.render(service_input['mode'],))
                        self.log.debug('Episode rendering for [{}] sent.'.format(service_input['mode']))

                    else:  # ignore any other input
                        # NOTE: response string must include 'ctrl' key
                        # for env.reset(), env.get_stat(), env.close() correct operation.
                        message = {'ctrl': 'send control keys: <_reset>, <_getstat>, <_render>, <_stop>.'}
                        self.log.debug('Server sent: ' + str(message))
                        socket.send_pyobj(message)  # pairs any other input

                else:
                    message = 'No <ctrl> key received:{}\nHint: forgot to call reset()?'.format(msg)
                    self.log.debug(message)
                    socket.send_pyobj(message)

            # Got '_reset' signal -> prepare Cerebro subclass and run episode:
            start_time = time.time()
            cerebro = copy.deepcopy(self.cerebro)
            cerebro._socket = socket
            cerebro._log = self.log
            cerebro._render = self.render
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
            data_server_response = self._comm_with_timeout(
                socket=data_socket,
                message={'ctrl': '_get_data'},
                timeout=self.connect_timeout,
            )
            if data_server_response['status'] in 'ok':
                self.log.debug('Data_server responded with datafeed in about {} seconds.'.
                               format(data_server_response['time']))

            else:
                msg = 'BtgymServer: data_server unreachable with status: <{}>.'. \
                    format(data_server_response['status'])
                self.log.error(msg)
                raise ConnectionError(msg)

            episode_datafeed = data_server_response['message']['datafeed']

            # Get episode data statistic and pass it to strategy params:
            cerebro.strats[0][0][2]['dataset_stat'] = data_server_response['message']['dataset_stat']
            cerebro.strats[0][0][2]['episode_stat'] = data_server_response['message']['episode_stat']

            # Set nice broker cash plotting:
            cerebro.broker.set_shortcash(False)

            # Add data to engine:
            cerebro.adddata(episode_datafeed)

            # Finally:
            episode = cerebro.run(stdstats=True, preload=False, oldbuysell=True)[0]

            # Update episode rendering:
            _ = self.render.render('just_render', cerebro=cerebro)
            _ = None

            # Recover that bloody analytics:
            analyzers_list = episode.analyzers.getnames()
            analyzers_list.remove('_env_analyzer')

            elapsed_time = timedelta(seconds=time.time() - start_time)
            self.log.info('Episode elapsed time: {}.'.format(elapsed_time))

            episode_result['episode'] = episode_number
            episode_result['runtime'] = elapsed_time
            episode_result['length'] = len(episode.data.close)

            for name in analyzers_list:
                episode_result[name] = episode.analyzers.getbyname(name).get_analysis()

            gc.collect()

        # Just in case -- we actually shouldn't get there except by some error:
        return None
