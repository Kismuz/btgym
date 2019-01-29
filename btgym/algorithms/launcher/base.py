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
#
# Original asynchronous framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#


import os
from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import time
import psutil
import glob
from subprocess import PIPE
import signal
import numpy as np
import copy

from btgym.algorithms.worker import Worker
from btgym.algorithms.aac import A3C
from btgym.algorithms.policy import BaseAacPolicy

import sys
sys.path.insert(0,'..')


class Launcher():
    """
    Configures and starts distributed TF training session with workers
    running sets of separate instances of BTgym/Atari environment.

    """

    def __init__(self,
                 env_config=None,
                 cluster_config=None,
                 policy_config=None,
                 trainer_config=None,
                 max_env_steps=None,
                 save_secs=600,
                 root_random_seed=None,
                 test_mode=False,
                 purge_previous=1,
                 render_last_env=True,
                 log_level=None,

                 verbose=0):
        """


        Args:
            env_config (dict):          environment class_config_dict, see 'Note' below.
            cluster_config (dict):      tf cluster configuration, see 'Note' below.
            policy_config (dict):       policy class_config_dict holding corr. policy class args.
            trainer_config (dict):      trainer class_config_dict holding corr. trainer class args.
            max_env_steps (int):        total number of environment steps to run training on.
            save_secs(int):             save model checkpoint every N secs.
            root_random_seed (int):     int or None
            test_mode (bool):           if True - use Atari gym env., BTGym otherwise.
            purge_previous (int):       keep or remove previous log files and saved checkpoints from log_dir:
                                        {0 - keep, 1 - ask, 2 - remove}.
            render_last_env:            bool, if True and there is more than one environment specified for each worker,
                                        only allows rendering for last environment in a list;
                                        allows rendering for all environments of a chief worker otherwise;
            verbose (int):              verbosity mode, {0 - WARNING, 1 - INFO, 2 - DEBUG}.
            log_level (int):            logbook level {DEBUG=10, INFO=11, NOTICE=12, WARNING=13},
                                        overrides `verbose` arg.

        Note:
            class_config_dict:  dictionary containing at least two keys:
                                - `class_ref`:    reference to class constructor or function;
                                - `kwargs`:       dictionary of keyword arguments, see corr. environment class args.

            cluster_config:     dictionary containing at least these keys:
                                - 'host':         cluster host, def: '127.0.0.1'
                                - 'port':         cluster port, def: 12222
                                - 'num_workers':  number of workers to run, def: 1
                                - 'num_ps':       number of parameter servers, def: 1
                                - 'num_envs':     number of environments to run in parallel for each worker, def: 1
                                - 'log_dir':      directory to save model and summaries, def: './tmp/btgym_aac_log'

        """

        self.env_config = dict(
            class_ref=None,
            kwargs=dict(
                port=5000,
                data_port=4999,
                gym_id=None,
            )
        )
        self.cluster_config = dict(
            host='127.0.0.1',
            port=12222,
            num_workers=1,
            num_ps=1,
            log_dir='./tmp/btgym_aac_log',
            initial_ckpt_dir=None,
            log_ckpt_subdir='/current_train_checkpoint',
            num_envs=1,
        )
        self.policy_config = dict(
            class_ref=BaseAacPolicy,
            kwargs=dict(
                lstm_layers=(256,)
            )
        )
        self.trainer_config = dict(
            class_ref=A3C,
            kwargs={}
        )
        self.ports_to_use = []
        self.root_random_seed = root_random_seed
        self.purge_previous = purge_previous
        self.test_mode = test_mode
        self.log_level = log_level
        self.verbose = verbose
        self.save_secs = save_secs
        self.render_last_env = render_last_env

        if max_env_steps is not None:
            self.max_env_steps = max_env_steps

        else:
            self.max_env_steps = 100 * 10 ** 6

        self.env_config = self._update_config_dict(self.env_config, env_config)

        self.cluster_config = self._update_config_dict(self.cluster_config, cluster_config)

        self.policy_config = self._update_config_dict(self.policy_config, policy_config)

        self.trainer_config = self._update_config_dict(self.trainer_config, trainer_config)

        self.trainer_config['kwargs']['test_mode'] = self.test_mode

        # Logging config:
        StreamHandler(sys.stdout).push_application()
        if self.log_level is None:
            log_levels = [(0, NOTICE), (1, INFO), (2, DEBUG)]
            self.log_level = WARNING
            for key, value in log_levels:
                if key == self.verbose:
                    self.log_level = value
        self.log = Logger('LauncherShell', level=self.log_level)

        # Seeding:
        if self.root_random_seed is not None:
            np.random.seed(self.root_random_seed)
            self.log.info('Random seed: {}'.format(self.root_random_seed))

        # Seeding for workers:
        self.workers_rnd_seeds = list(
            np.random.randint(0, 2 ** 30, self.cluster_config['num_workers'] + self.cluster_config['num_ps'])
        )

        # Log_dir housekeeping:
        if os.path.exists(self.cluster_config['log_dir']):
            # Remove previous log files and saved model checkpoints if opted:
            if self.purge_previous > 0:
                confirm = 'y'
                if self.purge_previous < 2:
                    confirm = input('<{}> already exists. Override[y/n]? '.format(self.cluster_config['log_dir']))
                if confirm in 'y':
                    files = glob.glob(self.cluster_config['log_dir'] + '/*')
                    p = psutil.Popen(['rm', '-R', ] + files, stdout=PIPE, stderr=PIPE)
                    self.log.notice('files in: {} purged.'.format(self.cluster_config['log_dir']))

            else:
                self.log.notice('writing to: {}.'.format(self.cluster_config['log_dir']))

        else:
            os.makedirs(self.cluster_config['log_dir'])
            self.log.notice('<{}> created.'.format(self.cluster_config['log_dir']))

        for kwarg in ['port', 'data_port']:
            assert kwarg in self.env_config['kwargs'].keys()

        assert self.env_config['class_ref'] is not None

        # Make cluster specification dict:
        self.cluster_spec = self._make_cluster_spec(self.cluster_config)

        # Configure workers:
        self.workers_config_list = self._make_workers_spec()

        # Ensure data_server port is clear:
        self.clear_port(self.env_config['kwargs']['data_port'])

        self.log.debug('Launcher ready.')

    def _make_workers_spec(self):
        """
        Creates list of workers specifications.
        Returns:
            list of dict
        """
        workers_config_list = []
        env_ports = np.arange(self.cluster_config['num_envs'], dtype=np.int32)
        env_data_ports = np.zeros(self.cluster_config['num_envs'], dtype=np.int32)
        worker_port = self.env_config['kwargs']['port']  # start value for BTGym comm. port

        # TODO: Hacky, cause dataset is threadlocked; do: pass dataset as class_ref + kwargs_dict:
        if self.test_mode:
            dataset_instance = None

        else:
            dataset_instance = self.env_config['kwargs'].pop('dataset')

        for key, spec_list in self.cluster_spec.items():
            task_index = 0  # referenced farther as worker id
            for _id in spec_list:
                env_config = copy.deepcopy(self.env_config)
                worker_config = {}
                if key in 'worker':
                    # Configure worker BTgym environment:
                    if task_index == 0:
                        env_config['kwargs']['data_master'] = True  # set worker_0 as chief and data_master
                        env_config['kwargs']['dataset'] = dataset_instance
                        env_config['kwargs']['render_enabled'] = True
                    else:
                        env_config['kwargs']['data_master'] = False
                        # env_config['kwargs']['dataset'] = dataset_instance
                        env_config['kwargs']['render_enabled'] = False  # disable rendering for all but chief

                    # Add list of connection ports for every parallel env for each worker:
                    env_config['kwargs']['port'] = list(worker_port + env_ports)
                    env_config['kwargs']['data_port'] = list(env_config['kwargs']['data_port'] + env_data_ports)
                    worker_port += self.cluster_config['num_envs']
                worker_config.update(
                    {
                        'env_config': env_config,
                        'policy_config': self.policy_config,
                        'trainer_config': self.trainer_config,
                        'cluster_spec': self.cluster_spec,
                        'job_name': key,
                        'task': task_index,
                        'test_mode': self.test_mode,
                        'log_dir': self.cluster_config['log_dir'],
                        'initial_ckpt_dir': self.cluster_config['initial_ckpt_dir'],
                        'log_ckpt_subdir': self.cluster_config['log_ckpt_subdir'],
                        'max_env_steps': self.max_env_steps,
                        'save_secs': self.save_secs,
                        'log_level': self.log_level,
                        'random_seed': self.workers_rnd_seeds.pop(),
                        'render_last_env': self.render_last_env
                    }
                )
                self.clear_port(env_config['kwargs']['port'])
                workers_config_list.append(worker_config)
                task_index += 1

        return workers_config_list

    def _make_cluster_spec(self, config):
        """
        Composes cluster specification dictionary.
        """
        cluster = {}
        all_ps = []
        port = config['port']

        for _ in range(config['num_ps']):
            self.clear_port(port)
            self.ports_to_use.append(port)
            all_ps.append('{}:{}'.format(config['host'], port))
            port += 1
        cluster['ps'] = all_ps

        all_workers = []
        for _ in range(config['num_workers']):
            self.clear_port(port)
            self.ports_to_use.append(port)
            all_workers.append('{}:{}'.format(config['host'], port))
            port += 1
        cluster['worker'] = all_workers
        return cluster

    def clear_port(self, port_list):
        """
        Kills process on specified ports list, if any.
        """
        if not isinstance(port_list, list):
            port_list = [port_list]

        for port in port_list:
            p = psutil.Popen(['lsof', '-i:{}'.format(port), '-t'], stdout=PIPE, stderr=PIPE)
            pid = p.communicate()[0].decode()[:-1]  # retrieving PID
            if pid is not '':
                p = psutil.Popen(['kill', pid])
                self.log.info('port {} cleared'.format(port))

    def _update_config_dict(self, old_dict, new_dict=None):
        """
        Service, updates nested dictionary with values from other one of same structure.

        Args:
            old_dict:   dict to update to
            new_dict:   dict to update from

        Returns:
            new updated dict
        """
        if type(new_dict) is not dict:
            new_dict = old_dict  # ~identity op

        for key, value in new_dict.items():
            if type(value) == dict:
                if key not in old_dict.keys():
                    old_dict[key] = {}
                old_dict[key] = self._update_config_dict(old_dict[key], value)

            else:
                old_dict[key] = value

        return old_dict

    def run(self):
        """
        Launches processes:

            distributed workers;
            parameter_server.
        """
        workers_list = []
        p_servers_list = []
        chief_worker = None

        def signal_handler(signal, frame):
            nonlocal workers_list
            nonlocal chief_worker
            nonlocal p_servers_list

            def stop_worker(worker_list):
                for worker in worker_list:
                    worker.terminate()

            stop_worker(workers_list)
            stop_worker([chief_worker])
            stop_worker(p_servers_list)

        # Start workers:
        for worker_config in self.workers_config_list:
            # Make:
            worker = Worker(**worker_config)
            # Launch:
            worker.daemon = False
            worker.start()

            if worker.job_name in 'worker':
                # Allow data-master to launch datafeed_server:
                if worker_config['env_config']['kwargs']['data_master']:
                    time.sleep(5)
                    chief_worker = worker

                else:
                    workers_list.append(worker)

            else:
                p_servers_list.append(worker)

        # TODO: auto-launch tensorboard?

        signal.signal(signal.SIGINT, signal_handler)

        # Halt here:
        msg = '\n********************************************************************************************\n' +\
                '**  Press `Ctrl-C` or jupyter:[Kernel]->[Interrupt] to stop training and close launcher.  **\n' + \
                '********************************************************************************************\n'
        print(msg)
        signal.pause()

        # Wait every worker to finish:
        for worker in workers_list:
            worker.join()
            self.log.notice('worker_{} has joined.'.format(worker.task))

        chief_worker.join()
        self.log.notice('chief_worker_{} has joined.'.format(chief_worker.task))

        for ps in p_servers_list:
            ps.terminate()
            ps.join()
            self.log.notice('parameter_server_{} has joined.'.format(ps.task))

        # TODO: close tensorboard
        # TODO: maybe export TB summaries accumulators links

        self.log.notice('Launcher closed.')

    def export_checkpoint(self, save_path):
        """
        Helper function: copies last saved checkpoint files to specified location;
        usually to serve as pre-trained model.

        Args:
            save_path:  path to copy checkpoint files to;

        """
        source = self.cluster_config['log_dir'] + self.cluster_config['log_ckpt_subdir']
        target = save_path

        assert os.path.exists(source), 'Source dir not found: {}'.format(source)

        if not os.path.exists(target):
            os.makedirs(target)
            self.log.notice('target dir created: {}'.format(target))

        else:
            old_files = glob.glob(target + '/*')
            p = psutil.Popen(['rm', '-R', ] + old_files, stdout=PIPE, stderr=PIPE)
            self.log.notice('target dir purged.')

        p = os.popen('cp -R ' + source + '/* ' + target)

        self.log.notice('copied to: {}'.format(target))








