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

from .worker import Worker
from .a3c import A3C
from .policy import BaseAacPolicy



class Launcher():
    """Configures and starts distributed TF training session with workers
    running separate instances of BTgym/Atari environment.

    """
    # "Register" legal kwargs:
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
    policy_class = BaseAacPolicy
    policy_config = dict(
        lstm_layers=(256,),
        pix_change=True,
    )
    trainer_class = A3C
    verbose = 0
    test_mode = False

    train_steps = None
    model_summary_freq = None
    episode_summary_freq = None
    env_render_freq = None
    model_gamma = None
    model_gae_lambda = None
    model_beta = None
    opt_learn_rate = None
    opt_end_learn_rate = None
    opt_decay_steps = None
    opt_decay = None
    opt_momentum = None
    opt_epsilon = None
    rollout_length = None
    pi_old_update_period = None
    num_epochs = None
    replay_memory_size = None
    replay_rollout_length = None
    use_off_policy_aac = None
    use_reward_prediction = None
    use_pixel_control = None
    use_value_replay = None
    use_rebalanced_replay = None
    rebalance_skewness = None
    off_aac_lambda = None
    rp_lambda = None
    pc_lambda = None
    vr_lambda = None
    gamma_pc = None
    rp_reward_threshold = None
    rp_sequence_size = None


    ports_to_use = []

    def __init__(self, root_random_seed=None, **kwargs):
        """

        Args:
            root_random_seed:   int, random seed.
            **kwargs:           passed to worker, trainer, environment runner.
        """
        self.root_random_seed = root_random_seed

        # Update attrs with kwargs:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if key in dir(self):
                self_value = getattr(self, key)
                # Partial dict attr update (only first level, no nested dict!):
                if type(self_value) == dict:
                    self_value.update(value)
                    setattr(self, key, self_value)
                else:
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

        if not self.test_mode:
            # We should have those to proceed with BTgym env.:
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
                        'policy_class': self.policy_class,
                        'policy_config': self.policy_config,
                        'trainer_class': self.trainer_class,
                        'cluster_spec': self.cluster_spec,
                        'job_name': key,
                        'task': task_index,
                        'log_dir': self.cluster_config['log_dir'],
                        'max_steps': self.train_steps,
                        'log': self.log,
                        'log_level': self.log.getEffectiveLevel(),
                        'random_seed': workers_rnd_seeds.pop()
                    }
                )
                self.clear_port(env_config['port'])
                self.workers_config_list.append(worker_config)
                task_index += 1

        self.clear_port(self.env_config['data_port'])
        self.log.debug('Launcher ready.')

    def make_cluster_spec(self, config):
        """Composes cluster specification dictionary.
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
        """Launches processes:
            tf distributed workers;
            tf parameter_server.
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
                if worker_config['env_config']['data_master']:
                    time.sleep(5)
                    chief_worker = worker

                else:
                    workers_list.append(worker)

            else:
                p_servers_list.append(worker)

        # TODO: launch tensorboard

        signal.signal(signal.SIGINT, signal_handler)

        # Halt here:
        msg = 'Press `Ctrl-C` to stop training and close launcher.'
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




