#
# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397

from logbook import Logger, StreamHandler
import sys
import os
import random
import multiprocessing

import tensorflow as tf

sys.path.insert(0,'..')
tf.logging.set_verbosity(tf.logging.INFO) # suppress tf.train.MonitoredTrainingSession deprecation warning
# TODO: switch to tf.train.MonitoredTrainingSession


class _FastSaver(tf.train.Saver):
    """
    Disables write_meta_graph argument,
    which freezes entire process and is mostly useless.
    """
    def save(self,
             sess,
             save_path,
             global_step=None,
             latest_filename=None,
             meta_graph_suffix="meta",
             write_meta_graph=True):
        super(_FastSaver, self).save(sess,
                                     save_path,
                                     global_step,
                                     latest_filename,
                                     meta_graph_suffix,
                                     False)


class Worker(multiprocessing.Process):
    """
    Distributed tf worker class.

    Sets up environment, trainer and starts training process in supervised session.
    """
    env_list = None

    def __init__(self,
                 env_config,
                 policy_config,
                 trainer_config,
                 cluster_spec,
                 job_name,
                 task,
                 log_dir,
                 log_level,
                 max_env_steps,
                 random_seed=None,
                 render_last_env=False,
                 test_mode=False):
        """

        Args:
            env_config:         environment class_config_dict.
            policy_config:      model policy estimator class_config_dict.
            trainer_config:     algorithm class_config_dict.
            cluster_spec:       tf.cluster specification.
            job_name:           worker or parameter server.
            task:               integer number, 0 is chief worker.
            log_dir:            for tb summaries and checkpoints.
            log_level:          int, logbook.level
            max_env_steps:      number of environment steps to run training on
            random_seed:        int or None
            render_last_env:    bool, if True - render enabled for last environment in a list; first otherwise
            test_mode:          if True - use Atari mode, BTGym otherwise.

            Note:
                - Conventional `self.global_step` refers to number of environment steps,
                    summarized over all environment instances, not to number of policy optimizer train steps.

                - Every worker can run several environments in parralell, as specified by `cluster_config'['num_envs'].
                    If use 4 forkers and num_envs=4 => total number of environments is 16. Every env instance has
                    it's own ThreadRunner process.

                - When using replay memory, keep in mind that every ThreadRunner is keeping it's own replay memory,
                    If memory_size = 2000, num_workers=4, num_envs=4 => total replay memory size equals 32 000 frames.
        """
        super(Worker, self).__init__()
        self.env_class = env_config['class_ref']
        self.env_kwargs = env_config['kwargs']
        self.policy_config = policy_config
        self.trainer_class = trainer_config['class_ref']
        self.trainer_kwargs = trainer_config['kwargs']
        self.cluster_spec = cluster_spec
        self.job_name = job_name
        self.task = task
        self.log_dir = log_dir
        self.max_env_steps = max_env_steps
        self.log_level = log_level
        self.log = None
        self.test_mode = test_mode
        self.random_seed = random_seed
        self.render_last_env = render_last_env

    def run(self):
        """Worker runtime body.
        """
        # Logging:
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('Worker_{}'.format(self.task), level=self.log_level)

        tf.reset_default_graph()

        if self.test_mode:
            import gym

        # Define cluster:
        cluster = tf.train.ClusterSpec(self.cluster_spec).as_cluster_def()

        # Start tf.server:
        if self.job_name in 'ps':
            server = tf.train.Server(
                cluster,
                job_name=self.job_name,
                task_index=self.task,
                config=tf.ConfigProto(device_filters=["/job:ps"])
            )
            self.log.debug('parameters_server started.')
            # Just block here:
            server.join()

        else:
            server = tf.train.Server(
                cluster,
                job_name='worker',
                task_index=self.task,
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=1,  # original was: 1
                    inter_op_parallelism_threads=2  # original was: 2
                )
            )
            self.log.debug('tf.server started.')

            self.log.debug('making environments:')
            # Making as many environments as many entries in env_config `port` list:
            # TODO: Hacky-II: only one example over all parallel environments can be data-master [and renderer]
            # TODO: measure data_server lags, maybe launch several instances
            self.env_list = []
            env_kwargs = self.env_kwargs.copy()
            env_kwargs['log_level'] = self.log_level
            port_list = env_kwargs.pop('port')
            data_port_list = env_kwargs.pop('data_port')
            data_master = env_kwargs.pop('data_master')
            render_enabled = env_kwargs.pop('render_enabled')

            render_list = [False for entry in port_list]
            if render_enabled:
                if self.render_last_env:
                    render_list[-1] = True
                else:
                    render_list[0] = True

            data_master_list = [False for entry in port_list]
            if data_master:
                data_master_list[0] = True

            # Parallel envs. numbering:
            if len(port_list) > 1:
                task_id = 0.0
            else:
                task_id = 0

            for port, data_port, is_render, is_master in zip(port_list, data_port_list, render_list, data_master_list):
                # Get random seed for environments:
                env_kwargs['random_seed'] = random.randint(0, 2 ** 30)

                if not self.test_mode:
                    # Assume BTgym env. class:
                    self.log.debug('setting env at port_{} is data_master: {}'.format(port, data_master))
                    self.log.debug('env_kwargs:')
                    for k, v in env_kwargs.items():
                        self.log.debug('{}: {}'.format(k, v))
                    try:
                        self.env_list.append(
                            self.env_class(
                                port=port,
                                data_port=data_port,
                                data_master=is_master,
                                render_enabled=is_render,
                                task=self.task + task_id,
                                **env_kwargs
                            )
                        )
                        data_master = False
                        self.log.info('set BTGym environment {} @ port:{}, data_port:{}'.
                                      format(self.task + task_id, port, data_port))
                        task_id += 0.01

                    except:
                        self.log.exception(
                            'failed to make BTGym environment at port_{}.'.format(port)
                        )
                        raise RuntimeError

                else:
                    # Assume atari testing:
                    try:
                        self.env_list.append(self.env_class(env_kwargs['gym_id']))
                        self.log.debug('set Gyn/Atari environment.')

                    except:
                        self.log.exception('failed to make Gym/Atari environment')
                        raise RuntimeError

            self.log.debug('Defining trainer...')

            # Define trainer:
            trainer = self.trainer_class(
                env=self.env_list,
                task=self.task,
                policy_config=self.policy_config,
                log_level=self.log_level,
                cluster_spec=self.cluster_spec,
                random_seed=self.random_seed,
                **self.trainer_kwargs,
            )

            self.log.debug('trainer ok.')

            # Saver-related:
            variables_to_save = [v for v in tf.global_variables() if not 'local' in v.name]
            local_variables = [v for v in tf.global_variables() if 'local' in v.name] + tf.local_variables()
            init_op = tf.variables_initializer(variables_to_save)
            local_init_op = tf.variables_initializer(local_variables)
            init_all_op = tf.global_variables_initializer()

            saver = _FastSaver(variables_to_save)

            self.log.debug('VARIABLES TO SAVE:')
            for v in variables_to_save:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            def init_fn(ses):
                self.log.info("initializing all parameters.")
                ses.run(init_all_op)

            config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])
            logdir = os.path.join(self.log_dir, 'train')
            summary_dir = logdir + "_{}".format(self.task)

            summary_writer = tf.summary.FileWriter(summary_dir)

            self.log.debug('before tf.train.Supervisor... ')

            # TODO: switch to tf.train.MonitoredTrainingSession
            sv = tf.train.Supervisor(
                is_chief=(self.task == 0),
                logdir=logdir,
                saver=saver,
                summary_op=None,
                init_op=init_op,
                local_init_op=local_init_op,
                init_fn=init_fn,
                #ready_op=tf.report_uninitialized_variables(variables_to_save),
                ready_op=tf.report_uninitialized_variables(),
                global_step=trainer.global_step,
                save_model_secs=300,
            )
            self.log.info("connecting to the parameter server... ")

            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                #sess.run(trainer.sync)
                trainer.start(sess, summary_writer)

                # Note: `self.global_step` refers to number of environment steps
                # summarized over all environment instances, not to number of policy optimizer train steps.
                global_step = sess.run(trainer.global_step)
                self.log.notice("started training at step: {}".format(global_step))

                while not sv.should_stop() and global_step < self.max_env_steps:
                    trainer.process(sess)
                    global_step = sess.run(trainer.global_step)

                # Ask for all the services to stop:
                for env in self.env_list:
                    env.close()

                sv.stop()
            self.log.notice('reached {} steps, exiting.'.format(global_step))



