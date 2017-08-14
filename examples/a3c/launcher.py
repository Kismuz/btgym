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


import sys
sys.path.insert(0,'..')

import os
import logging
import time
import psutil
from subprocess import PIPE
#import multiprocessing

#import tensorflow as tf

from worker import Worker


class Launcher():
    """
    Starts distributed TF training session with workers running separate BTgym environments.
    """
    env_class = None
    env_config = dict(
        port=5000,
        data_port=4999,
        gym_id=None,
    )
    cluster_config = dict(
        host='127.0.0.1',
        port=12222,
        num_workers=1,
        num_ps=1,
        log_dir='./tmp/a3c_log',
    )
    verbose = 0
    test_mode = False

    # Legal kwargs:
    train_steps = None
    model_summary_freq = None
    episode_summary_freq = None
    env_render_freq = None
    model_gamma = None
    model_lambda = None
    model_learn_rate = None
    rollout_length= None

    ports_to_use = []

    def __init__(self, **kwargs):
        """_____"""
        # Update attrs with kwargs:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if key in dir(self):
                # TODO: partial dict attr update
                setattr(self, key, value)
            else:
                raise KeyError('Unexpected key argument: {}={}'.format(key, value))

        # Logging config:
        logging.basicConfig()
        self.log = logging.getLogger('Launcher')
        log_levels = [(0, 'WARNING'), (1, 'INFO'), (2, 'DEBUG'),]
        for key, level in log_levels:
            if key == self.verbose:
                self.log.setLevel(level)

        if not os.path.exists(self.cluster_config['log_dir']):
            os.makedirs(self.cluster_config['log_dir'])
            self.log.info('{} created.'.format(self.cluster_config['log_dir']))

        if not self.test_mode:
            # We should have theese to proceed with BTgym env.:
            assert 'port' in self.env_config.keys()
            assert 'data_port' in self.env_config.keys()
            assert self.env_class is not None

        # Make cluster specification dict:
        self.cluster_spec = self.make_cluster_spec(self.cluster_config)

        # Configure workers:
        self.workers_config_list = []

        for key, spec_list in self.cluster_spec.items():
            task_index = 0
            for worker_id in spec_list:
                env_config = dict()
                worker_config = dict()
                worker_config.update(self.kwargs)
                env_config.update(self.env_config)
                if key in 'worker':
                    # Configure  worker BTgym environment:
                    if task_index == 0:
                        env_config['data_master'] = True  # set worker_0 as chief and data_master
                    else:
                        env_config['data_master'] = False
                        env_config['port'] += task_index  # increment connection port
                        env_config['render_enabled'] = False  # disable rendering for all but chief

                worker_config.update(
                    {
                        'env_class': self.env_class,
                        'env_config': env_config,
                        'cluster_spec': self.cluster_spec,
                        'job_name': key,
                        'task': task_index,
                        'log_dir': self.cluster_config['log_dir'],
                        'max_steps': self.train_steps,
                        'log': self.log
                    }
                )
                self.workers_config_list.append(worker_config)
                task_index += 1

        self.log.debug('Launcher ready.')

    def make_cluster_spec(self, config):
        """
        Composes cluster specification dictionary. Clears ports to use btw.
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

    def run(self):
        """
        Launches processes:
            tf distributed workers;
            tf parameter_server
        """
        workers_list = []
        p_servers_list = []

        # Start workers:
        for worker_config in self.workers_config_list:
            # Make:
            worker = Worker(**worker_config)
            # Launch:
            worker.daemon = False
            worker.start()

            if worker.job_name in 'worker':
                # Allow data-master to launch datafeed_server:
                if worker_config['env_config']['data_master']:
                    time.sleep(5)
                    chief_worker = worker

                else:
                    workers_list.append(worker)

            else:
                p_servers_list.append(worker)


        # TODO: launch tensorboard

        # Wait every worker to finish:
        for worker in workers_list:
            worker.join()
            self.log.info('worker_{} has joined.'.format(worker.task))

        chief_worker.join()
        self.log.info('chief_worker_{} has joined.'.format(chief_worker.task))

        # Kill param_servers:
        for ps in p_servers_list:
            ps.terminate()
            ps.join()
            self.log.info('parameter_server_{} has joined.'.format(ps.task))

        # TODO: close tensorboard

        self.log.info('Launcher closed.')




