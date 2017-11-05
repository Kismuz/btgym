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



import sys
sys.path.insert(0,'..')

import os
import logging
import time
import psutil
from subprocess import PIPE
import signal
import numpy as np
import copy

from .worker import Worker
from .a3c import A3C
from .policy import BaseAacPolicy



class Launcher():
    """
    Configures and starts distributed TF training session with workers
    running separate instances of BTgym/Atari environment.

    """

    def __init__(self,
                 env_config=None,
                 cluster_config=None,
                 policy_config=None,
                 trainer_config=None,
                 max_train_steps=None,
                 root_random_seed=None,
                 test_mode=False,
                 verbose=0):
        """


        Args:
            env_config:         environment class_config_dict (see 'Note' below)
            cluster_config:     dictionary containing keys: 'host', 'port', 'num_workers', 'num_ps'=1, 'log_dir'
            policy_config:      policy class_config_dict
            trainer_config:     trainer class_config_dict
            max_train_steps:    number of train steps to run
            root_random_seed:   int or None
            test_mode:          if True - use Atari gym env., BTGym otherwise.
            verbose:            0 - silent, 1 - info, 3 - debug level

        Note:
            class_config_dict:  dictionary containing at least two keys:
                                    `class_ref`: reference to class constructor or function;
                                    `kwargs`: dictionary of keyword arguments passed to `class_ref`
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
        self.max_train_steps = 10 * 10 ** 6
        self.ports_to_use = []
        self.root_random_seed = root_random_seed
        self.test_mode = test_mode
        self.verbose = verbose

        if max_train_steps is not None:
            self.max_train_steps = max_train_steps

        self.env_config = self.update_config_dict(self.env_config, env_config)

        self.cluster_config = self.update_config_dict(self.cluster_config, cluster_config)

        self.policy_config = self.update_config_dict(self.policy_config, policy_config)

        self.trainer_config = self.update_config_dict(self.trainer_config, trainer_config)

        self.trainer_config['kwargs']['test_mode'] = self.test_mode

        # Logging config:
        logging.basicConfig()
        self.log = logging.getLogger('Launcher')
        log_levels = [(0, 'WARNING'), (1, 'INFO'), (2, 'DEBUG'),]
        for key, level in log_levels:
            if key == self.verbose:
                self.log.setLevel(level)

        # Seeding:
        if self.root_random_seed is not None:
            np.random.seed(self.root_random_seed)
        self.log.info('Random seed: {}'.format(self.root_random_seed))

        # Seeding for workers:
        workers_rnd_seeds = list(
            np.random.randint(0, 2**30, self.cluster_config['num_workers'] + self.cluster_config['num_ps'])
        )

        if not os.path.exists(self.cluster_config['log_dir']):
            os.makedirs(self.cluster_config['log_dir'])
            self.log.info('{} created.'.format(self.cluster_config['log_dir']))

        #if not self.test_mode:
        # We should have those to proceed with BTgym workers configuration:
        for kwarg in ['port', 'data_port']:
            assert kwarg in self.env_config['kwargs'].keys()

        assert self.env_config['class_ref'] is not None

        # Make cluster specification dict:
        self.cluster_spec = self.make_cluster_spec(self.cluster_config)

        # Configure workers:
        self.workers_config_list = []

        for key, spec_list in self.cluster_spec.items():
            task_index = 0
            for worker_id in spec_list:
                env_config = copy.deepcopy(self.env_config)
                worker_config = {}
                if key in 'worker':
                    # Configure  worker BTgym environment:
                    if task_index == 0:
                        env_config['kwargs']['data_master'] = True  # set worker_0 as chief and data_master
                    else:
                        env_config['kwargs']['data_master'] = False
                        env_config['kwargs']['port'] += task_index  # increment connection port

                        env_config['kwargs']['render_enabled'] = False  # disable rendering for all but chief
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
                        'max_train_steps': self.max_train_steps,
                        'log': self.log,
                        'log_level': self.log.getEffectiveLevel(),
                        'random_seed': workers_rnd_seeds.pop()
                    }
                )
                self.clear_port(env_config['kwargs']['port'])
                self.workers_config_list.append(worker_config)
                task_index += 1

        self.clear_port(self.env_config['kwargs']['data_port'])
        self.log.debug('Launcher ready.')

    def make_cluster_spec(self, config):
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

    def clear_port(self, port):
        """
        Kills process on specified port, if any.
        """
        p = psutil.Popen(['lsof', '-i:{}'.format(port), '-t'], stdout=PIPE, stderr=PIPE)
        pid = p.communicate()[0].decode()[:-1]  # retrieving PID
        if pid is not '':
            p = psutil.Popen(['kill', pid])
            self.log.info('port {} cleared'.format(port))

    def update_config_dict(self, old_dict, new_dict=None):
        """
        Updates nested dictionary with values from other one of same structure.

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
                old_dict[key] = self.update_config_dict(old_dict[key], value)

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

        # TODO: launch tensorboard

        signal.signal(signal.SIGINT, signal_handler)

        # Halt here:
        msg = 'Press `Ctrl-C` or [Kernel]->[Interrupt] to stop training and close launcher.'
        print(msg)
        self.log.info(msg)
        signal.pause()

        # Wait every worker to finish:
        for worker in workers_list:
            worker.join()
            self.log.info('worker_{} has joined.'.format(worker.task))

        chief_worker.join()
        self.log.info('chief_worker_{} has joined.'.format(chief_worker.task))

        for ps in p_servers_list:
            ps.join()
            self.log.info('parameter_server_{} has joined.'.format(ps.task))

        # TODO: close tensorboard

        self.log.info('Launcher closed.')




