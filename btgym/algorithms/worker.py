#
# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397

import sys
sys.path.insert(0,'..')

import os
import logging
import multiprocessing

import cv2
import tensorflow as tf


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
                 log,
                 log_level,
                 max_env_steps,
                 random_seed=None,
                 test_mode=False):
        """

        Args:
            env_config:     environment class_config_dict.
            policy_config:  model policy estimator class_config_dict.
            trainer_config: algorithm class_config_dict.
            cluster_spec:   tf.cluster specification.
            job_name:       worker or parameter server.
            task:           integer number, 0 is chief worker.
            log_dir:        for tb summaries and checkpoints.
            log:            parent logger
            log_level:      0 - silent, 1 - info, 3 - debug level
            max_env_steps:  number of environment steps to run training on
            test_mode:      if True - use Atari mode, BTGym otherwise.

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
        self.log = log
        self.log_level = log_level
        logging.basicConfig()
        self.log = logging.getLogger('{}_{}'.format(self.job_name, self.task))
        self.log.setLevel(log_level)
        self.test_mode = test_mode
        self.random_seed = random_seed

    def run(self):
        """Worker runtime body.
        """
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
            self.log.debug('worker_{} tf.server started.'.format(self.task))

            self.log.debug('making environment.')
            # Making as many environments as many entries in env_config `port` list:
            self.env_list = []
            env_kwargs = self.env_kwargs.copy()
            env_kwargs['log'] = self.log
            port_list = env_kwargs.pop('port')

            for port in port_list:
                if not self.test_mode:
                    # Assume BTgym env. class:
                    self.log.debug('worker_{} is data_master: {}'.format(self.task, self.env_kwargs['data_master']))
                    try:
                        self.env_list.append(self.env_class(port=port, **env_kwargs))

                    except:
                        raise SystemExit(' Worker_{} failed to make BTgym environment'.format(self.task))

                else:
                    # Assume atari testing:
                    try:
                        self.env_list.append(self.env_class(env_kwargs['gym_id']))

                    except:
                        raise SystemExit(' Worker_{} failed to make Gym environment'.format(self.task))

            self.log.debug('worker_{}:envronment ok.'.format(self.task))

            # Define trainer:
            trainer = self.trainer_class(
                env=self.env_list,
                task=self.task,
                policy_config=self.policy_config,
                log=self.log,
                random_seed=self.random_seed,
                **self.trainer_kwargs,
            )

            self.log.debug('worker_{}:trainer ok.'.format(self.task))

            # Saver-related:
            variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
            init_op = tf.variables_initializer(variables_to_save)
            init_all_op = tf.global_variables_initializer()

            saver = _FastSaver(variables_to_save)

            self.log.debug('worker_{}: vars_to_save:'.format(self.task))
            for v in variables_to_save:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            def init_fn(ses):
                self.log.debug("Initializing all parameters.")
                ses.run(init_all_op)

            config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])
            logdir = os.path.join(self.log_dir, 'train')
            summary_dir = logdir + "_{}".format(self.task)

            summary_writer = tf.summary.FileWriter(summary_dir)

            sv = tf.train.Supervisor(
                is_chief=(self.task == 0),
                logdir=logdir,
                saver=saver,
                summary_op=None,
                init_op=init_op,
                init_fn=init_fn,
                ready_op=tf.report_uninitialized_variables(variables_to_save),
                global_step=trainer.global_step,
                save_model_secs=300,
            )
            self.log.debug("worker_{}: connecting to the parameter server... ".format(self.task))

            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                sess.run(trainer.sync)
                trainer.start(sess, summary_writer)
                # Note: `self.global_step` refers to number of environment steps
                # summarized over all environment instances, not to number of policy optimizer train steps.
                global_step = sess.run(trainer.global_step)
                # Fill replay memory, if any: TODO: remove
                if hasattr(trainer,'memory'):
                    if not trainer.memory.is_full():
                        trainer.memory.fill()

                self.log.warning("worker_{}: started training at step: {}".format(self.task, global_step))
                while not sv.should_stop() and global_step < self.max_env_steps:
                    trainer.process(sess)
                    global_step = sess.run(trainer.global_step)

                # Ask for all the services to stop:
                for env in self.env_list:
                    env.close()

                sv.stop()
            self.log.warning('worker_{}: reached {} steps, exiting.'.format(self.task, global_step))



