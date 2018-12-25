
import numpy as np
import copy

from btgym.algorithms.launcher.base import Launcher


class MetaLauncher(Launcher):
    """
    Launcher class with extended functionality to support gradient-based meta-learning algorithms.
    For every distributed worker it configures two master/slave environments such that that slave environment
    always runs same data trial as master one.
    Typically master environment is configured to run episodes from train data of the trial and salve one - from test
    data. With AAC framework properly set up it enables single worker to estimate meta-loss by collecting relevant
    test and train trajectories in parallel.
    """

    def __init__(self, cluster_config=None, render_slave_env=True, **kwargs):
        """

        Args:
            cluster_config:     environment class_config_dict
            render_slave_env:   bool, if True - rendering enabled for slave environment; master otherwise.
            **kwargs:           same as base class args: btgym.algorithms.launcher.Launcher
        """
        meta_cluster_config = dict(
            host='127.0.0.1',
            port=12222,
            num_workers=1,
            num_ps=1,
            log_dir='./tmp/meta_aac_log',
            num_envs=2,  # just a declaration
        )
        meta_cluster_config = self._update_config_dict(meta_cluster_config, cluster_config)

        # Force number of parallel envs anyway:
        meta_cluster_config['num_envs'] = 2

        self.render_slave_env = render_slave_env

        # Update:
        kwargs['cluster_config'] = meta_cluster_config
        kwargs['test_mode'] = False

        super(MetaLauncher, self).__init__(**kwargs)

    def _make_workers_spec(self):
        """
        Creates list of workers specifications.
        Overrides base class method. Sets master/slave pair of environments for every worker.

        Returns:
            list of dict
        """
        workers_config_list = []
        env_ports = np.arange(self.cluster_config['num_envs'], dtype=np.int32)
        env_data_ports = np.zeros(self.cluster_config['num_envs'], dtype=np.int32)
        worker_port = self.env_config['kwargs']['port']  # start value for BTGym comm. port

        # TODO: Hacky, cause dataset is threadlocked; do: pass dataset as class_ref + kwargs_dict:
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
                        env_config['kwargs']['render_enabled'] = False  # disable rendering for all but chief

                    # Add list of connection ports for every parallel env for each worker:
                    env_config['kwargs']['port'] = list(worker_port + env_ports)
                    # Here master/slave pair is defined:
                    env_config['kwargs']['data_port'] = [
                        env_config['kwargs']['data_port'],  # data_server_port
                        env_config['kwargs']['port'][0]     # comm_port of master env. as data_port for slave
                    ]

                    self.log.info('env_config: {}'.format(env_config))

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
                        'max_env_steps': self.max_env_steps,
                        'log_level': self.log_level,
                        'random_seed': self.workers_rnd_seeds.pop(),
                        'render_last_env': self.render_slave_env  # last env in a pair is slave
                    }
                )
                self.clear_port(env_config['kwargs']['port'])
                workers_config_list.append(worker_config)
                task_index += 1

        return workers_config_list
