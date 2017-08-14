# This code borrows heavily from OpenAI universal_starter_agent:
# https://github.com/openai/universe-starter-agent
# Under MIT licence.

import cv2
import go_vncdriver

import sys
sys.path.insert(0,'..')

import os
import logging
import multiprocessing

import IPython.display as Display
import PIL.Image as Image

import tensorflow as tf

from a3c import A3C
from envs import create_env

class FastSaver(tf.train.Saver):
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
        super(FastSaver, self).save(sess,
                                    save_path,
                                    global_step,
                                    latest_filename,
                                    meta_graph_suffix,
                                    False)

class Worker(multiprocessing.Process):
    """___"""

    def __init__(self,
                 env_class,
                 env_config,
                 cluster_spec,
                 job_name,
                 task,
                 log_dir,
                 log,
                 max_steps=100000000,
                 test_mode=False,
                 **kwargs):
        """___"""
        super(Worker, self).__init__()
        self.env_class = env_class
        self.env_config = env_config
        self.cluster_spec = cluster_spec
        self.job_name = job_name
        self.task = task
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.log = log
        logging.basicConfig()
        self.log = logging.getLogger('{}_{}'.format(self.job_name, self.task))
        self.log.setLevel('DEBUG')
        self.kwargs = kwargs
        self.test_mode = test_mode

    def show_rendered_image(self, rgb_array):
        """
        Convert numpy array to RGB image using PILLOW and
        show it inline using IPykernel.
        """
        Display.display(Image.fromarray(rgb_array))

    def render_all_modes(self):
        """
        Retrieve and show environment renderings
        for all supported modes.
        """
        for mode in self.env.metadata['render.modes']:
            print('[{}] mode:'.format(mode))
            self.show_rendered_image(self.env.render(mode))

    def run(self):
        """
        Worker runtime body.
        """

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
            self.log.info('parameters_server started.')
            # Just block here:
            server.join()

        else:
            server = tf.train.Server(
                cluster,
                job_name='worker',
                task_index=self.task,
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=2
                )
            )
            self.log.debug('tf.server started.')

            self.log.debug('making environment.')
            if not self.test_mode:
                # Assume BTgym env. class:
                self.log.debug('worker_{} is data_master: {}'.format(self.task, self.env_config['data_master']))
                try:
                    self.env = self.env_class(**self.env_config)

                except:
                    raise SystemExit(' Worker_{} failed to make BTgym environment'.format(self.task))

            else:
                # Assume atari testing:
                try:
                    self.env = create_env(self.env_config['gym_id'])

                except:
                    raise SystemExit(' Worker_{} failed to make Atari Gym environment'.format(self.task))

            # Define trainer:
            trainer = A3C(env=self.env, task=self.task, test_mode=self.test_mode, **self.kwargs)

            # Saver-related:
            variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
            init_op = tf.variables_initializer(variables_to_save)
            init_all_op = tf.global_variables_initializer()

            saver = FastSaver(variables_to_save)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            #self.log.debug('worker-{}: trainable vars:'.format(self.task))
            #for v in var_list:
            #    self.log.debug('{}: {}'.format(v.name, v.get_shape()))

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
                #summary_writer=summary_writer,  # TODO do we need it here?
                ready_op=tf.report_uninitialized_variables(variables_to_save),
                global_step=trainer.global_step,
                save_model_secs=300,
            )
            self.log.debug("connecting to the parameter server... ")

            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                sess.run(trainer.sync)
                self.log.debug('worker_{}: trainer synch`ed'.format(self.task))
                trainer.start(sess, summary_writer)
                self.log.debug('worker_{}: trainer started'.format(self.task))
                global_step = sess.run(trainer.global_step)
                self.log.info("worker_{}: starting training at step: {}".format(self.task, global_step))
                while not sv.should_stop() and global_step < self.max_steps:
                    trainer.process(sess)
                    global_step = sess.run(trainer.global_step)

                    # TEST:

                    if False:
                        print('RENDER ATTEMPT at {}...'.format(global_step))
                        self.render_all_modes()
                        print('RENDERED')

            # Ask for all the services to stop:
            sv.stop()
            self.env.close()
            self.log.info('worker_{}: reached {} steps, exiting.'.format(self.task, global_step))




class TestTrainer():
    """Dummy trainer class."""
    global_step = 0

    def __init__(self, worker_id):
        self.worker_id = worker_id

    def start(self):
        print('Trainer_{} started.'.format(self.worker_id))

    def sync(self):
        print('Trainer_{}: sync`ed.'.format(self.worker_id))

    def process(self):
        print('Traner_{}: processed step {}'.format(self.worker_id, self.global_step))
        self.global_step += 1

