
from btgym.algorithms import BaseAAC


import tensorflow as tf
import  numpy as np


class GA3C_0_0(BaseAAC):
    """
    Guided policy search framework to be.
    """
    def get_sample_config(self):
        """
        Experimental.
        Stage 1:
            Constantly keep sampling from same trial.
            How fast guide policy converges?
            Simple cheap FF policy for gude?

        """
        #sess = tf.get_default_session()

        # request new trial every `self.num_train_episodes`:

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
            self.log.info('Local policy sync at {}-th local train step'.format(self.local_steps))
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