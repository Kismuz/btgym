###############################################################################
#
# Copyright (C) 2017-19 Andrew Muzikin
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
import gc

import itertools
import zmq
import copy

import time, datetime
import random
from datetime import timedelta

import backtrader as bt
from .datafeed import DataSampleConfig, EnvResetConfig
from .strategy.observers import NormPnL, Position, Reward

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
        self.data_socket = self.strategy.env._data_socket
        self.render = self.strategy.env._render

        # Pass data serving methods:
        self.get_current_trial = self.strategy.env._get_data
        self.can_broadcast = self.strategy.can_broadcast
        self.get_timestamp = self.strategy._get_timestamp
        self.get_dataset_info = self.strategy.env._get_info
        self.get_broadcast_info = self.strategy._get_broadcast_info

        self.message = None
        self.step_to_render = None  # Due to reset(), this will get populated before first render() call.
        self.respond_pending = False

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

    def send_env_response(self, is_done):
        """
        Sends environment response as <o, r, d, i> tuple.
        See issue #84.
        """
        # Gather response:
        raw_state = self.strategy.get_raw_state()
        state = self.strategy.get_state()
        reward = self.strategy.get_reward()
        # Send response as <o, r, d, i> tuple (Gym convention),
        # opt to send entire info_list or just latest part:
        info = [self.info_list[-1]]
        self.socket.send_pyobj((state, reward, is_done, info))

        # Increment global time by sending timestamp to data_server, if authorized;
        if self.can_broadcast:
            global_timestamp = self.get_timestamp()
            broadcast_info = self.get_broadcast_info()
            self.log.debug('broadcasting timestamp: {}'.format(global_timestamp))

            self.data_socket.send_pyobj(
                {
                    'ctrl': '_set_broadcast_message',
                    'timestamp': global_timestamp,
                    'broadcast_message': broadcast_info,
                }
            )
            broadcast_set_response = self.data_socket.recv_pyobj()
            self.log.debug('DATA_COMM/broadcast received: {}'.format(broadcast_set_response))

        # Back up step information for rendering.
        # It pays when using skip-frames: will'll get future state otherwise.

        self.step_to_render = ({'human': raw_state}, state, reward, is_done, self.info_list)

        # Reset info:
        self.info_list = []
        self.strategy.env_iteration += 1
        self.respond_pending = False

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
        self.strategy.action = self.strategy.p.initial_portfolio_action
        # Trick to avoid excessive orders emitting during skip_frame loop:
        self.strategy.action['_skip_this'] = True

        # Only if it's time to communicate or episode has come to end:
        if self.strategy.iteration % self.strategy.p.skip_frame == 0 or is_done:
            if self.respond_pending:
                # Other side is waiting for response:
                self.send_env_response(is_done)

                # If done, initiate fallback to Control Mode:
                if is_done:
                    self.early_stop()
                    return

            #print('Analyzer_strat_iteration:', self.strategy.iteration)
            #print('Analyzer_env_iteration:', self.strategy.env_iteration)

            # Halt and wait to receive message from outer world:
            self.message = self.socket.recv_pyobj()
            msg = 'COMM received: {}'.format(self.message)
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
                elif self.message['ctrl'] == '_done':
                    is_done = True  # redundant
                    self.socket.send_pyobj('_DONE SIGNAL RECEIVED')
                    self.early_stop()
                    return None

                elif self.message['ctrl'] == '_get_data':
                    self.socket.send_pyobj(self.get_current_trial())

                elif self.message['ctrl'] == '_get_info':
                    self.socket.send_pyobj(self.get_dataset_info())

                # Unknown key:
                else:
                    message = {'ctrl': 'send control keys: <_reset>, <_getstat>, ' +
                                       '<_render>, <_stop>, or valid agent action'}
                    self.log.warning(
                        'Analyzer received unexpected key: {}; Sent: {}'.format(self.message, str(message))
                    )
                    self.socket.send_pyobj(message)

                # Halt again:
                self.message = self.socket.recv_pyobj()
                msg = 'COMM recieved: {}'.format(self.message)
                self.log.debug(msg)

            # Store agent action an rise respond_pending flag:
            if 'action' in self.message:  # now it should!
                self.strategy.action = self.message['action']
                self.strategy.last_action = self.message['action']
                self.respond_pending = True

            else:
                msg = 'No <action> key recieved:\n' + msg
                raise AssertionError(msg)

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

        dict(action=<control action, type=str>,), where control action is:
        '_reset' - rewinds backtrader engine and runs new episode;
        '_getstat' - retrieve episode results and statistics;
        '_stop' - server shut-down.

    Control mode OUT::

        <string message> - reports current server status;
        <statisic dict> - last run episode statisics.  NotImplemented.

        Within-episode signals:
        Episode mode IN:
        dict(action=<agent_action, type=str>,), where agent_action is:
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

    def __init__(
        self,
        cerebro=None,
        render=None,
        network_address=None,
        data_network_address=None,
        connect_timeout=90,
        log_level=None,
        task=0,
    ):
        """

        Args:
            cerebro:                backtrader.cerebro engine class.
            render:                 render class
            network_address:        environmnet communication, str
            data_network_address:   data communication, str
            connect_timeout:        seconds, int
            log_level:              int, logbook.level
        """

        super(BTgymServer, self).__init__()
        self.task = task
        self.log_level = log_level
        self.log = None
        self.process = None
        self.cerebro = cerebro
        self.network_address = network_address
        self.render = render
        self.data_network_address = data_network_address
        self.connect_timeout = connect_timeout # server connection timeout in seconds.
        self.connect_timeout_step = 0.01

        self.trial_sample = None
        self.trial_stat = None
        self.dataset_stat = None

    @staticmethod
    def _comm_with_timeout(socket, message):
        """
        Exchanges messages via socket with timeout.

        Note:
            socket zmq.RCVTIMEO and zmq.SNDTIMEO should be set to some finite number of milliseconds

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

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                response['status'] = 'send_failed_due_to_connect_timeout'

            else:
                response['status'] = 'send_failed_for_unknown_reason'
            return response

        start = time.time()
        try:
            response['message'] = socket.recv_pyobj()
            response['time'] =  time.time() - start

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                response['status'] = 'receive_failed_due_to_connect_timeout'

            else:
                response['status'] = 'receive_failed_for_unknown_reason'
            return response

        return response

    def get_dataset_stat(self):
        data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': '_get_info'}
        )
        if data_server_response['status'] in 'ok':
            self.log.debug('Data_server @{} responded with dataset statistic in about {} seconds.'.
                           format(self.data_network_address, data_server_response['time']))

            return data_server_response['message']

        else:
            msg = 'BtgymServer_sampling_attempt: data_server @{} unreachable with status: <{}>.'. \
                format(self.data_network_address, data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

    def get_trial(self, **reset_kwargs):
        """

        Args:
            reset_kwargs:   dictionary of args to pass to parent data iterator

        Returns:
            trial_sample, trial_stat, dataset_stat
        """
        wait = 0
        while True:
            # Get new data subset:
            data_server_response = self._comm_with_timeout(
                socket=self.data_socket,
                message={'ctrl': '_get_data', 'kwargs': reset_kwargs}
            )
            if data_server_response['status'] in 'ok':
                self.log.debug('Data_server @{} responded in ~{:1.6f} seconds.'.
                               format(self.data_network_address, data_server_response['time']))

            else:
                msg = 'BtgymServer_sampling_attempt: data_server @{} unreachable with status: <{}>.'. \
                    format(self.data_network_address, data_server_response['status'])
                self.log.error(msg)
                raise ConnectionError(msg)

            # Ready or not?
            try:
                assert 'Dataset not ready' in data_server_response['message']['ctrl']
                if wait <= self.wait_for_data_reset:
                    pause = random.random() * 2
                    time.sleep(pause)
                    wait += pause
                    self.log.info(
                        'Domain dataset not ready, wait time left: {:4.2f}s.'.format(self.wait_for_data_reset - wait)
                    )
                else:
                    data_server_response = self._comm_with_timeout(
                        socket=self.data_socket,
                        message={'ctrl': '_stop'}
                    )
                    self.socket.close()
                    self.context.destroy()
                    raise RuntimeError('Failed to assert Domain dataset is ready. Exiting.')

            except (AssertionError, KeyError) as e:
                break
        # Get trial instance:
        trial_sample = data_server_response['message']['sample']
        trial_stat = trial_sample.describe()
        trial_sample.reset()
        dataset_stat = data_server_response['message']['stat']
        origin = data_server_response['message']['origin']
        timestamp = data_server_response['message']['timestamp']

        return trial_sample, trial_stat, dataset_stat, origin, timestamp

    def get_trial_message(self):
        """
        Prepares  message containing current trial instance, mimicking data_server message protocol.
        Intended for serving requests from data_slave environment.

        Returns:
            dict containing trial instance, d_set statistic and origin key; dict containing 'ctrl' response if master
            d_set is not ready;
        """
        if self.trial_sample is not None:
            message = {
                'sample': self.trial_sample,
                'stat': self.dataset_stat,
                'origin': 'master_environment',
                'timestamp': self.get_global_time()
            }

        else:
            message = {'ctrl': 'Dataset not ready, hold on...'}
            self.log.debug('Sent to slave: ' + str(message))

        return message

    def get_global_time(self):
        """
        Asks dataserver for current dataset global_time.

        Returns:
            POSIX timestamp
        """
        data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': '_get_global_time'}
        )
        if data_server_response['status'] in 'ok':
            pass

        else:
            msg = 'BtgymServer_sampling_attempt: data_server @{} unreachable with status: <{}>.'. \
                format(self.data_network_address, data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        return data_server_response['message']['timestamp']

    def get_broadcast_message(self):
        """
        Asks dataserver for current dataset global_time and broadcast message.

        Returns:
            POSIX timestamp
        """
        data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': '_get_broadcast_message'}
        )
        if data_server_response['status'] in 'ok':
            pass

        else:
            msg = 'BtgymServer_sampling_attempt: data_server @{} unreachable with status: <{}>.'. \
                format(self.data_network_address, data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        return data_server_response['message']['timestamp'], data_server_response['message']['broadcast_message']

    def run(self):
        """
        Server process runtime body. This method is invoked by env._start_server().
        """
        # Logging:
        from logbook import Logger, StreamHandler, WARNING
        import sys
        StreamHandler(sys.stdout).push_application()
        if self.log_level is None:
            self.log_level = WARNING
        self.log = Logger('BTgymServer_{}'.format(self.task), level=self.log_level)

        self.process = multiprocessing.current_process()
        self.log.info('PID: {}'.format(self.process.pid))

        # Runtime Housekeeping:
        cerebro = None
        episode_result = dict()
        episode_sample = None

        # How long to wait for data_master to reset data:
        self.wait_for_data_reset = 300  # seconds

        connect_timeout = 60  # in seconds

        # Set up a comm. channel for server as ZMQ socket
        # to carry both service and data signal
        # !! Reminder: Since we use REQ/REP - messages do go in pairs !!
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, -1)
        self.socket.setsockopt(zmq.SNDTIMEO, connect_timeout * 1000)
        self.socket.bind(self.network_address)

        self.data_context = zmq.Context()
        self.data_socket = self.data_context.socket(zmq.REQ)
        self.data_socket.setsockopt(zmq.RCVTIMEO, connect_timeout * 1000)
        self.data_socket.setsockopt(zmq.SNDTIMEO, connect_timeout * 1000)
        self.data_socket.connect(self.data_network_address)

        # Check connection:
        self.log.debug('Pinging data_server at: {} ...'.format(self.data_network_address))

        data_server_response = self._comm_with_timeout(
            socket=self.data_socket,
            message={'ctrl': 'ping!'}
        )
        if data_server_response['status'] in 'ok':
            self.log.debug('Data_server seems ready with response: <{}>'.
                          format(data_server_response['message']))

        else:
            msg = 'Data_server unreachable with status: <{}>.'.\
                format(data_server_response['status'])
            self.log.error(msg)
            raise ConnectionError(msg)

        # Init renderer:
        self.render.initialize_pyplot()

        # Mandatory DrawDown and auxillary plotting observers to add to data-master strategy instance:
        # TODO: make plotters optional args
        if self.render.enabled:
            aux_obsrevers = [bt.observers.DrawDown, Reward, Position, NormPnL]

        else:
            aux_obsrevers = [bt.observers.DrawDown]

        # Server 'Control Mode' loop:
        for episode_number in itertools.count(0):
            while True:
                # Stuck here until '_reset' or '_stop':
                service_input = self.socket.recv_pyobj()
                msg = 'Control mode: received <{}>'.format(service_input)
                self.log.debug(msg)

                if 'ctrl' in service_input:
                    # It's time to exit:
                    if service_input['ctrl'] == '_stop':
                        # Server shutdown logic:
                        # send last run statistic, release comm channel and exit:
                        message = 'Exiting.'
                        self.log.info(message)
                        self.socket.send_pyobj(message)
                        self.socket.close()
                        self.context.destroy()
                        return None

                    # Start episode:
                    elif service_input['ctrl'] == '_reset':
                        message = 'Preparing new episode with kwargs: {}'.format(service_input['kwargs'])
                        self.log.debug(message)
                        self.socket.send_pyobj(message)  # pairs '_reset'
                        break

                    # Retrieve statistic:
                    elif service_input['ctrl'] == '_getstat':
                        self.socket.send_pyobj(episode_result)
                        self.log.debug('Episode statistic sent.')

                    # Send episode rendering:
                    elif service_input['ctrl'] == '_render' and 'mode' in service_input.keys():
                        # Just send what we got:
                        self.socket.send_pyobj(self.render.render(service_input['mode']))
                        self.log.debug('Episode rendering for [{}] sent.'.format(service_input['mode']))

                    # Serve data-dependent environment with trial instance:
                    elif service_input['ctrl'] == '_get_data':
                        message = 'Sending trial data to slave'
                        self.log.debug(message)
                        self.socket.send_pyobj(self.get_trial_message())

                    # Serve data-dependent environment with dataset statisitc:
                    elif service_input['ctrl'] == '_get_info':
                        message = 'Sending dataset statistic to slave'
                        self.log.debug(message)
                        self.socket.send_pyobj(self.get_dataset_stat())

                    else:  # ignore any other input
                        # NOTE: response string must include 'ctrl' key
                        # for env.reset(), env.get_stat(), env.close() correct operation.
                        message = {'ctrl': 'send control keys: <_reset>, <_getstat>, <_render>, <_stop>.'}
                        self.log.debug('Control mode: sent: ' + str(message))
                        self.socket.send_pyobj(message)  # pairs any other input

                else:
                    message = 'No <ctrl> key received:{}\nHint: forgot to call reset()?'.format(msg)
                    self.log.debug(message)
                    self.socket.send_pyobj(message)

            # Got '_reset' signal -> prepare Cerebro subclass and run episode:
            start_time = time.time()
            cerebro = copy.deepcopy(self.cerebro)
            cerebro._socket = self.socket
            cerebro._data_socket = self.data_socket
            cerebro._log = self.log
            cerebro._render = self.render

            # Pass methods for serving capabilities:
            cerebro._get_data = self.get_trial_message
            cerebro._get_info = self.get_dataset_stat

            # Add auxillary observers, if not already:
            for aux in aux_obsrevers:
                is_added = False
                for observer in cerebro.observers:
                    if aux in observer:
                        is_added = True
                if not is_added:
                    cerebro.addobserver(aux)

            # Add communication utility:
            cerebro.addanalyzer(_BTgymAnalyzer, _name='_env_analyzer',)

            # Data preparation:

            # Renew system state:
            current_timestamp, current_broadcast_message = self.get_broadcast_message()

            # Parse args we got with _reset call:
            sample_config = dict(
                episode_config=copy.deepcopy(DataSampleConfig),
                trial_config=copy.deepcopy(DataSampleConfig)
            )
            for key, config in sample_config.items():
                try:
                    config.update(service_input['kwargs'][key])

                except KeyError:
                    self.log.debug(
                        '_reset <{}> kwarg not found, using default values: {}'.format(key, config)
                    )
            sample_config['trial_config']['broadcast_message'] = current_broadcast_message
            sample_config['episode_config']['broadcast_message'] = current_broadcast_message

            # Get new Trial from data_server if requested,
            # despite bult-in new/reuse data object sampling option, perform checks here to avoid
            # redundant traffic:
            if sample_config['trial_config']['get_new'] or self.trial_sample is None:
                self.log.info(
                    'Requesting new Trial sample with args: {}'.format(sample_config['trial_config'])
                )
                self.trial_sample, self.trial_stat, self.dataset_stat, origin, current_timestamp =\
                    self.get_trial(**sample_config['trial_config'])

                if origin in 'data_server':
                    self.trial_sample.set_logger(self.log_level, self.task)

                self.log.debug('Got new Trial: <{}>'.format(self.trial_sample.filename))

            else:
                self.log.info('Reusing Trial <{}>'.format(self.trial_sample.filename))
                # current_timestamp = self.get_global_time()

            self.log.debug(
                'current global_time: {}'.format(datetime.datetime.fromtimestamp(current_timestamp))
            )
            # Get episode:
            if sample_config['episode_config']['timestamp'] is None or\
                    sample_config['episode_config']['timestamp'] < current_timestamp:
                sample_config['episode_config']['timestamp'] = current_timestamp

            self.log.info(
                'Requesting episode from <{}> with args: {}'.
                format(self.trial_sample.filename, sample_config['episode_config'])
            )

            episode_sample = self.trial_sample.sample(**sample_config['episode_config'])
            self.log.debug('Got new Episode: <{}>'.format(episode_sample.filename))

            # Get episode data statistic and pass it to strategy params:
            cerebro.strats[0][0][2]['trial_stat'] = self.trial_stat
            cerebro.strats[0][0][2]['trial_metadata'] = self.trial_sample.metadata
            cerebro.strats[0][0][2]['dataset_stat'] = self.dataset_stat
            cerebro.strats[0][0][2]['episode_stat'] = episode_sample.describe()
            cerebro.strats[0][0][2]['metadata'] = episode_sample.metadata

            cerebro.strats[0][0][2]['broadcast_message'] = current_broadcast_message

            # Set nice broker cash plotting:
            cerebro.broker.set_shortcash(False)

            # Convert and add data to engine:
            feed = episode_sample.to_btfeed()
            if isinstance(feed, dict):
                for key, stream in feed.items():
                    cerebro.adddata(stream, name=key)

            else:
                cerebro.adddata(feed, name='base_asset')

            # Finally:
            episode = cerebro.run(stdstats=True, preload=False, oldbuysell=True, tradehistory=True)[0]

            self.log.debug('Episode run finished.')

            # Update episode rendering:
            _ = self.render.render('just_render', cerebro=cerebro)
            _ = None

            # Recover that bloody analytics:
            analyzers_list = episode.analyzers.getnames()
            analyzers_list.remove('_env_analyzer')

            elapsed_time = timedelta(seconds=time.time() - start_time)
            self.log.debug('Episode elapsed time: {}.'.format(elapsed_time))

            episode_result['episode'] = episode_number
            episode_result['runtime'] = elapsed_time
            episode_result['length'] = len(episode.data.close)

            for name in analyzers_list:
                episode_result[name] = episode.analyzers.getbyname(name).get_analysis()

            gc.collect()

        # Just in case -- we actually shouldn't get there except by some error:
        return None
