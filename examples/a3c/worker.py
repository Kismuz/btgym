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
#import psutil
#from subprocess import PIPE
import multiprocessing

import tensorflow as tf


class Worker(multiprocessing.Process):
    """___"""
    global_step = 0

    def __init__(self,
                 env_class,
                 env_config,
                 cluster_spec,
                 job_name,
                 task,
                 log_dir,
                 max_steps,
                 log):
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
        self.log.setLevel('INFO')

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
            self.log.info('tf.server started.')

            self.log.info('making environment.')
            if self.env_class is not None:
                # Assume BTgym:
                self.log.debug('data_master: {}'.format(self.env_config['data_master']))
                try:
                    env = self.env_class(**self.env_config)

                except:
                    raise SystemExit(' Worker_{} failed to make environment'.format(self.task))

            # Define trainer:
            trainer = TestTrainer(self.task)

            train_op = lambda x: print('worker_{}: train_step: {}'.format(self.task, x))

            config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])
            logdir = os.path.join(self.log_dir, 'train')
            summary_dir = logdir + "_{}".format(self.task)

            summary_writer = tf.summary.FileWriter(summary_dir)

            sv = tf.train.Supervisor(
                is_chief=(self.task == 0),
                logdir=logdir,
                # saver=saver,
                summary_op=None,
                # init_op=init_op,
                # init_fn=init_fn,
                summary_writer=summary_writer,
                # ready_op=tf.report_uninitialized_variables(variables_to_save),
                # global_step=trainer.global_step,
                save_model_secs=60,
                save_summaries_secs=60,
            )

            self.log.debug("connecting to the parameter server... ")

            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                trainer.sync()
                trainer.start()
                global_step = trainer.global_step
                self.log.info("Starting training at step=%d", global_step)
                while not sv.should_stop() and global_step < self.max_steps:
                    trainer.process()
                    global_step = trainer.global_step

            # Ask for all the services to stop:
            sv.stop()
            env.close()
            self.log.info('reached {} steps, exiting.'.format(global_step))


class TestTrainer():
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

