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
import datetime

import tensorflow as tf

sys.path.insert(0, '..')
tf.logging.set_verbosity(tf.logging.INFO)


class FastSaver(tf.train.Saver):
    """
    Disables write_meta_graph argument,
    which freezes entire process and is mostly useless.
    """
    def save(
        self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False
    ):
        super(FastSaver, self).save(
            sess,
            save_path,
            global_step,
            latest_filename,
            meta_graph_suffix,
            write_meta_graph=False,
        )


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
                 log_ckpt_subdir,
                 initial_ckpt_dir,
                 save_secs,
                 log_level,
                 max_env_steps,
                 random_seed=None,
                 render_last_env=True,
                 test_mode=False):
        """

        Args:
            env_config:             environment class_config_dict.
            policy_config:          model policy estimator class_config_dict.
            trainer_config:         algorithm class_config_dict.
            cluster_spec:           tf.cluster specification.
            job_name:               worker or parameter server.
            task:                   integer number, 0 is chief worker.
            log_dir:                path for tb summaries and current checkpoints.
            log_ckpt_subdir:        log_dir subdirectory to store current checkpoints
            initial_ckpt_dir:       path for checkpoint to load as pre-trained model.
            save_secs:              int, save model checkpoint every N secs.
            log_level:              int, logbook.level
            max_env_steps:          number of environment steps to run training on
            random_seed:            int or None
            render_last_env:        bool, if True and there is more than one environment specified for each worker,
                                    only allows rendering for last environment in a list;
                                    allows rendering for all environments of a chief worker otherwise;
            test_mode:              if True - use Atari mode, BTGym otherwise.

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
        self.is_chief = (self.task == 0)
        self.log_dir = log_dir
        self.save_secs = save_secs
        self.max_env_steps = max_env_steps
        self.log_level = log_level
        self.log = None
        self.test_mode = test_mode
        self.random_seed = random_seed
        self.render_last_env = render_last_env

        # Saver and summaries path:
        self.current_ckpt_dir = self.log_dir + log_ckpt_subdir
        self.initial_ckpt_dir = initial_ckpt_dir
        self.summary_dir = self.log_dir + '/worker_{}'.format(self.task)

        # print(log_ckpt_subdir)
        # print(self.log_dir)
        # print(self.current_ckpt_dir)
        # print(self.initial_ckpt_dir)
        # print(self.summary_dir)

        self.summary_writer = None
        self.config = None
        self.saver = None

    def _restore_model_params(self, sess, save_path):
        """
        Restores model parameters from specified location.

        Args:
            sess:       tf.Session obj.
            save_path:  path where parameters were previously saved.

        Returns: True if model has been successfully loaded, False otherwise.
        """
        if save_path is None:
            return False

        assert self.saver is not None, 'FastSaver has not been configured.'

        try:
            # Look for valid checkpoint:
            ckpt_state = tf.train.get_checkpoint_state(save_path)
            if ckpt_state is not None and ckpt_state.model_checkpoint_path:
                self.saver.restore(sess, ckpt_state.model_checkpoint_path)

            else:
                self.log.notice('no saved model parameters found in:\n{}'.format(save_path))
                return False

        except (ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
            self.log.notice('failed to restore model parameters from:\n{}'.format(save_path))
            return False

        return True

    def _save_model_params(self, sess, global_step):
        """
        Saves model checkpoint to predefined location.

        Args:
            sess:           tf.Session obj.
            global_step:    global step number is appended to save_path to create the checkpoint filenames
        """
        assert self.saver is not None, 'FastSaver has not been configured.'
        self.saver.save(
            sess,
            save_path=self.current_ckpt_dir + '/model_parameters',
            global_step=global_step
        )

    def run(self):
        """Worker runtime body.
        """
        # Logging:
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('Worker_{}'.format(self.task), level=self.log_level)
        try:
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
                        intra_op_parallelism_threads=4,  # original was: 1
                        inter_op_parallelism_threads=4,  # original was: 2
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
                        render_list = [True for entry in port_list]
                        # render_list[0] = True

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

                        except Exception as e:
                            self.log.exception(
                                'failed to make BTGym environment at port_{}.'.format(port)
                            )
                            raise e

                    else:
                        # Assume atari testing:
                        try:
                            self.env_list.append(self.env_class(env_kwargs['gym_id']))
                            self.log.debug('set Gyn/Atari environment.')

                        except Exception as e:
                            self.log.exception('failed to make Gym/Atari environment')
                            raise e

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
                init_op = tf.initializers.variables(variables_to_save)
                local_init_op = tf.initializers.variables(local_variables)
                init_all_op = tf.global_variables_initializer()

                def init_fn(_sess):
                    self.log.notice("initializing all parameters...")
                    _sess.run(init_all_op)

                # def init_fn_scaff(scaffold, _sess):
                #     self.log.notice("initializing all parameters...")
                #     _sess.run(init_all_op)

                # self.log.warning('VARIABLES TO SAVE:')
                # for v in variables_to_save:
                #     self.log.warning(v)
                #
                # self.log.warning('LOCAL VARS:')
                # for v in local_variables:
                #     self.log.warning(v)

                self.saver = FastSaver(var_list=variables_to_save, max_to_keep=1, save_relative_paths=True)

                self.config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])

                sess_manager = tf.train.SessionManager(
                    local_init_op=local_init_op,
                    ready_op=None,
                    ready_for_local_init_op=tf.report_uninitialized_variables(variables_to_save),
                    graph=None,
                    recovery_wait_secs=90,
                )
                with sess_manager.prepare_session(
                    master=server.target,
                    init_op=init_op,
                    config=self.config,
                    init_fn=init_fn,
                ) as sess:

                    # Try to restore pre-trained model
                    pre_trained_restored = self._restore_model_params(sess, self.initial_ckpt_dir)
                    _ = sess.run(trainer.reset_global_step)

                    if not pre_trained_restored:
                        # If not - try to recover current checkpoint:
                        current_restored = self._restore_model_params(sess, self.current_ckpt_dir)

                    else:
                        current_restored = False

                    if not pre_trained_restored and not current_restored:
                        self.log.notice('training from scratch...')

                    self.log.info("connecting to the parameter server... ")

                    self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
                    trainer.start(sess, self.summary_writer)

                    # Note: `self.global_step` refers to number of environment steps
                    # summarized over all environment instances, not to number of policy optimizer train steps.
                    global_step = sess.run(trainer.global_step)
                    self.log.notice("started training at step: {}".format(global_step))

                    last_saved_time = datetime.datetime.now()
                    last_saved_step = global_step

                    while global_step < self.max_env_steps:
                        trainer.process(sess)
                        global_step = sess.run(trainer.global_step)

                        time_delta = datetime.datetime.now() - last_saved_time
                        if self.is_chief and time_delta.total_seconds() > self.save_secs:
                            self._save_model_params(sess, global_step)
                            train_speed = (global_step - last_saved_step) / (time_delta.total_seconds() + 1)
                            self.log.notice(
                                'env. step: {}; cluster speed: {:.0f} step/sec; checkpoint saved.'.format(
                                    global_step,
                                    train_speed
                                )
                            )
                            last_saved_time = datetime.datetime.now()
                            last_saved_step = global_step

                # Ask for all the services to stop:
                for env in self.env_list:
                    env.close()

                self.log.notice('reached {} steps, exiting.'.format(global_step))

        except Exception as e:
            self.log.exception(e)
            raise e



