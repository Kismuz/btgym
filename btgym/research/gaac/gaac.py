
from btgym.algorithms import BaseAAC


import tensorflow as tf
import  numpy as np


class GA3C_0_0(BaseAAC):
    """
    Guided policy search framework to be.
    """
    def __init__(self, global_update_period=1, **kwargs):
        super(GA3C_0_0, self).__init__(_log_name='GAAC_0.0', **kwargs)

        # Local training:
        try:
            self.global_update_period = global_update_period
            with tf.device(self.worker_device):
                #self.local_optimizer = tf.train.AdamOptimizer(self.opt_learn_rate, epsilon=1e-5)
                local_grads_and_vars = list(zip(self.grads, self.local_network.var_list))
                self.local_train_op = self.optimizer.apply_gradients(local_grads_and_vars)
                #self.local_initializer = tf.variables_initializer(self.local_network.var_list) #TODO: WTF?
                #self.local_initializer = tf.global_variables_initializer()

        except:
            msg = 'Child class __init()__ exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def start(self, sess, summary_writer, **kwargs):
        """
        Executes all initializing operations,
        starts environment runner[s].
        Supposed to be called by parent worker just before training loop starts.

        Args:
            sess:           tf session object.
            kwargs:         not used by default.
        """
        try:
            # Initialize local:
            #sess.run(self.local_initializer)
            # Copy weights from global to local:
            sess.run(self.sync)

            # Start thread_runners:
            self._start_runners(sess, summary_writer)

        except:
            msg = 'Start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_sample_config(self):
        """
        Experimental.
        Stage 1:
            Trains on its own:
                Constantly keep sampling from same trial x(num_train_episodes).
                Updates from parameter server on new trial only
                How fast guide policy converges?
                Simple cheap FF policy for gude?

        """
        #sess = tf.get_default_session()

        # request new trial every `self.num_train_episodes`, replay old one otherwise:

        if self.current_train_episode < self.num_train_episodes:
            episode_type = 0  # train
            self.current_train_episode += 1
            new_trial = False

        else:
            # cycle end, reset and start new (rec. depth 1)
            self.current_train_episode = 0
            self.current_test_episode = 0
            episode_type = 0
            new_trial = True

            sess = tf.get_default_session()
            self.log.notice('Got new sample, local policy synced at {}-th local train step'.format(self.local_steps))
            sess.run(self.sync)

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=new_trial,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=new_trial,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config

    def process(self, sess):
        """
        Overrides default:
            Local training. Global model updated at every 50th train step.
        """
        try:
            # Collect data from child thread runners:
            data = self._get_data()

            # Copy weights from local policy to local target policy:
            if self.use_target_policy and self.local_steps % self.pi_prime_update_period == 0:
                sess.run(self.sync_pi_prime)

            # Test or train: if at least one on-policy rollout from parallel runners is test one -
            # set learn rate to zero for entire minibatch. Doh.
            try:
                is_train = not np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any()

            except KeyError:
                is_train = True

            #if is_train:
            #    # If there is no any test rollouts  - copy weights from shared to local new_policy:
            #    sess.run(self.sync_pi)

            # self.log.debug('is_train: {}'.format(is_train))

            feed_dict = self.process_data(sess, data, is_train)

            # Say No to redundant summaries:
            wirte_model_summary =\
                self.local_steps % self.model_summary_freq == 0

            # Train locally:
            fetches = [self.local_train_op]

            if self.local_steps % self.global_update_period == 0:
                fetches += [self.train_op]
                self.log.info('Global update at {}-th local train step'.format(self.local_steps))

            if wirte_model_summary:
                fetches_last = fetches + [self.model_summary_op, self.inc_step]
            else:
                fetches_last = fetches + [self.inc_step]

            # Do a number of SGD train epochs:
            # When doing more than one epoch, we actually use only last summary:
            for i in range(self.num_epochs - 1):
                fetched = sess.run(fetches, feed_dict=feed_dict)

            fetched = sess.run(fetches_last, feed_dict=feed_dict)

            if wirte_model_summary:
                model_summary = fetched[-2]

            else:
                model_summary = None

            # Write down summaries:
            self.process_summary(sess, data, model_summary)

            self.local_steps += 1

        except:
            msg = 'Train step exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

