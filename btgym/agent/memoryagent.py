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
import os
import numpy as np
import tensorflow as tf


class BTgymMemoryAgent():
    """
    DDQN/DRQN Agent for episodic tasks with
    multimodal state observation,
    discrete action space
    and sequential/random access replay memory.

    Bi-modal observation state shape:
    state_shape = dict(external=(N1,N2, ..., Nk),
                       internal=(M1, M2, ..., Ml),)

    Shape of single experience:
    experience_shape = dict(state_external=state_shape['external'],
                            state_internal=state_shape['internal'],
                            action=(),
                            reward=(),
                            state_internal_next=state_shape['internal'])
                            !->state_external_next is not stored, because of ...== state_external[i+1]

    """

    def __init__(
        self,
        estimator,
        state_shape,
        num_actions,
        max_episode_length,
        replay_memory_size=500000,
        replay_memory_init_size=50000,
        epsilon_start=0.99,
        epsilon_end=0.1,
        epsilon_decay_steps=500000,
        gamma=0.99,
        tau=0.001,
        batch_size=32,
        load_latest_checkpoint=False,
        home_dir='./btgym_agent_home/',
        scope='agent_smith',
    ):
        """______"""
        self.state_shape = state_shape
        self.experience_shape = dict(
            state_external=state_shape['external'],
            state_internal=state_shape['internal'],
            action=(),
            reward=(),
            state_internal_next=state_shape['internal'],
        )
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma

        self.num_actions = num_actions
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.replay_memory_init_size = replay_memory_init_size
        self.memory_shape = (round(replay_memory_size / max_episode_length), self.max_episode_length)

        self.local_step = 0  # step within current episode
        self.episode = 0 # keep track of eisode numbers within current tf.Session()
        self.current_mem_size = 0  # savable
        self.current_mem_pointer = -1  # savable

        # Save/restore housekeeping:
        self.saver = tf.train.Saver()
        self.home_dir = home_dir
        self.load_latest_checkpoint = load_latest_checkpoint

        self.action = None
        self.reward = None

        # Create directories for checkpoints and summaries:
        self.checkpoint_dir = os.path.join(self.home_dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.home_dir, "model")
        self.monitor_path = os.path.join(self.home_dir, "monitor")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.monitor_path):
            os.makedirs(self.monitor_path)

        # Create linear epsilon decay: epsilon =  a * x + b
        self.epsilon = lambda x: self.epsilon_end if x > self.epsilon_decay_steps else\
            (self.epsilon_end - self.epsilon_start) / self.epsilon_decay_steps * x + self.epsilon_start

        with tf.variable_scope(scope):
            # Mandatory:
            self._build_memory()
            self._build_model_updater(self.q_estimator, self.target_estimator, self.tau)
            self._build_global_step()

            # May or may not be here:
            self.q_estimator = estimator(state_shape=self.state_shape,
                                         scope='q')
            self.target_estimator = estimator(state_shape=self.state_shape,
                                              scope='target_q')

        # Make e-greedy policy function for estimator:
        self.ploicy = self.make_epsilon_greedy_policy(self.q_estimator, self.num_actions)

    def save(self, sess):
        """
        Saves current agent state.
        """
        self.saver.save(tf.get_default_session(), self.checkpoint_path)

    def restore(self, sess):
        """
        Restores agent state from latest saved checkpoint if it exists.
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint and self.load_latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)

    def _build_global_step(self):
        """
        Savable global_step constructor.
        """
        self._global_step = tf.Variable(
            0,
            name='estimator_train_step',
            trainable=False,
            dtype=tf.int32,
        )
        self.increment_global_step_op = tf.assign_add(self._global_step, 1)

    def global_step(self, sess):
        """
        Returns current step value.
        """
        return sess.run(self._global_step)

    def global_step_up(self, sess):
        """
        Increments global step count by 1.
        """
        sess.run(self.increment_global_step_op)

    def _build_memory(self):
        """
        Replay memory constructor.
        """
        # Runtime buffer to store experiences of single episode, doesn't get saved.
        self.episode_buffer = dict(
            state_external=np.zeros((self.max_episode_length,) + self.experience_shape['state_external']),
            state_internal=np.zeros((self.max_episode_length,) + self.experience_shape['state_internal']),
            action=np.zeros((self.max_episode_length,) + self.experience_shape['action']),
            reward=np.zeros((self.max_episode_length,) + self.experience_shape['reward']),
            state_internal_next=np.zeros((self.max_episode_length,) + \
                                         self.experience_shape['state_internal_next']),
            episode_length=np.zeros((1,)),
        )
        with tf.variable_scope('memory'):
            # Placeholders to feed single episode to memory tf.variables:
            self._mem_current_size_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(1,),
            )
            self._mem_cyclic_pointer_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(1,),
            )
            self._mem_pointer_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(1,),
            )
            self._mem_pointer2_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(1,),
            )
            self.indices1_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(None,),
            )
            self.indices2_pl = tf.placeholder(
                dtype=tf.int32,
                # shape=(None,),
            )

            self.mem_pl = dict(
                state_external=tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.max_episode_length,) + self.experience_shape['state_external'],
                ),
                state_internal=tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.max_episode_length,) + self.experience_shape['state_internal'],
                ),
                action=tf.placeholder(
                    dtype=tf.int32,
                    shape=(self.max_episode_length,) + self.experience_shape['action'],
                ),
                reward=tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.max_episode_length,) + self.experience_shape['reward'],
                ),
                state_internal_next=tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.max_episode_length,) + self.experience_shape['state_internal_next'],
                ),
                episode_length=tf.placeholder(
                    dtype=tf.int32,
                    shape=(1,),
                ),
            )
            # Savable memory variables:
            self._mem_current_size = tf.Variable(  # in number of episodes
                0,
                trainable=False,
                name='currnet_size',
                dtype=tf.int32,
            )
            self._mem_cyclic_pointer = tf.Variable(
                0,
                trainable=False,
                name='cyclic_pointer',
                dtype=tf.int32,
            )

            # Memory itself:
            self.memory = dict(
                state_external=tf.Variable(
                    tf.zeros(self.memory_shape + self.experience_shape['state_external']),
                    trainable=False,
                    name='state_external',
                    dtype=tf.float32,
                ),
                state_internal=tf.Variable(
                    tf.zeros(self.memory_shape + self.experience_shape['state_internal']),
                    trainable=False,
                    name='state_internal',
                    dtype=tf.float32,
                ),
                action=tf.Variable(
                    tf.zeros(self.memory_shape + self.experience_shape['action'], dtype=tf.int32),
                    trainable=False,
                    name='action',
                    dtype=tf.int32,
                ),
                reward=tf.Variable(
                    tf.zeros(self.memory_shape + self.experience_shape['reward']),
                    trainable=False,
                    name='reward',
                    dtype=tf.float32,
                ),
                state_internal_next=tf.Variable(
                    tf.zeros(self.memory_shape + self.experience_shape['state_internal_next']),
                    trainable=False,
                    name='state_internal_next',
                    dtype=tf.float32,
                ),
                episode_length=tf.Variable(
                    tf.zeros((self.memory_shape[0], 1), dtype=tf.int32),
                    trainable=False,
                    name='episode_length',
                    dtype=tf.int32,
                ),
            )

        # Build relevant operations:

        # Set memory pointers:
        self.set_cyclic_pointer_op = self._mem_cyclic_pointer.assign(self._mem_cyclic_pointer_pl),
        self.set_mem_current_size_op = self._mem_current_size.assign(self._mem_current_size_pl),

        # Store single episode in memory:
        self.save_episode_op = []
        for key, var in self.memory.items():
            op = var[self._mem_cyclic_pointer, ...].assign(self.mem_pl[key])
            self.save_episode_op.append(op)

        # Get single episode from memory:
        self.get_episode_op = []
        episode_length = self.memory['episode_length'][self._mem_pointer_pl]
        for key, var in self.memory.items():
            op = var[self._mem_pointer_pl, 0: episode_length[0], ...]
            self.get_episode_op.append(op)

        # Gather episode's length values by its numbers:
        self.get_episodes_length_op = tf.gather(self.memory['episode_length'], self.indices1_pl)

        # Get single experience from memory:
        self.get_experience_op = []
        for key, var in self.memory.items():
            if key != 'episode_length':
                op = var[self._mem_pointer_pl, self._mem_pointer2_pl, ...]
                self.get_experience_op.append(op)

    def get_memory_size(self, sess):
        """
        Returns current used memory size
        in number of stored episodes, in range [0, max_mem_size].
        """
        return sess.run(self._mem_current_size)

    def set_memory_size(self, sess, value):
        """
        Sets current used memory size in number of episodes.
        """
        assert value <= self.memory_shape[0]
        _ = sess.run(self.set_mem_current_size_op,
                     feed_dict={
                         self._mem_current_size_pl: value,
                     }
                     )

    def get_cyclic_pointer(self, sess):
        """
        Cyclic pointer stores number (==address) of episode in replay memory,
        currently to be written/replaced.
        This pointer supposed to infinitely loop through entire memory, updating records.
        Returns: current pointer value.
        """
        return sess.run(self._mem_cyclic_pointer)

    def set_cyclic_pointer(self, sess, value):
        """Sets one."""
        _ = sess.run(self.set_cyclic_pointer_op,
                     feed_dict={self._mem_cyclic_pointer_pl: value},
                     )

    def add_experience(self,
                       sess,
                       state,
                       action,
                       reward,
                       state_next,
                       done, ):
        """
        Writes single experience to episode memory buffer,
        appends episode to replay memory if episode is over.
        """
        # Append experience to buffer:
        self.episode_buffer['state_external'][self.local_step, ...] = state['external']
        self.episode_buffer['state_internal'][self.local_step, ...] = state['internal']
        self.episode_buffer['action'][self.local_step, ...] = action
        self.episode_buffer['reward'][self.local_step, ...] = reward
        self.episode_buffer['state_internal_next'][self.local_step, ...] = state_next['internal']
        self.episode_buffer['episode_length'][0] = self.local_step
        self.local_step += 1

        if done or self.local_step >= self.memory_shape[1]:
            # If over, write episode to replay memory:

            # Prepare feed dict:
            feeder = dict()
            for key, value in self.episode_buffer.items():
                feeder.update({self.mem_pl[key]: value})

            # Save:
            sess.run(
                self.save_episode_op,
                feed_dict=feeder,
            )

            # print('Saved episode with:')
            # print('mem_size:', self.get_memory_size(sess))
            # print('cyclic_pointer:', self.get_cyclic_pointer(sess))
            # Reset episode buffer, local_step:
            self.episode_buffer = dict(
                state_external=np.zeros((self.max_episode_length,) + self.experience_shape['state_external']),
                state_internal=np.zeros((self.max_episode_length,) + self.experience_shape['state_internal']),
                action=np.zeros((self.max_episode_length,) + self.experience_shape['action']),
                reward=np.zeros((self.max_episode_length,) + self.experience_shape['reward']),
                state_internal_next=np.zeros((self.max_episode_length,) + \
                                             self.experience_shape['state_internal_next']),
                episode_length=np.zeros((1,)),
            )
            self.local_step = 0
            self.episode += 1

            # Increment memory size and move cycle_pointer to next episode:
            # Get actual size and pointer:
            self.current_mem_size = self.get_memory_size(sess)
            self.current_mem_pointer = self.get_cyclic_pointer(sess)

            if self.current_mem_size < self.memory_shape[0] - 1:
                # If memory is not full - increase used size by 1,
                # else - leave it along:
                self.current_mem_size += 1
                self.set_memory_size(sess, self.current_mem_size)

            if self.current_mem_pointer >= self.current_mem_size:
                # Rewind cyclic pointer, if reached memory upper bound:
                self.set_cyclic_pointer(sess, 0)
                self.current_mem_pointer = 0

            else:
                # Step by one:
                self.current_mem_pointer += 1
                self.set_cyclic_pointer(sess, self.current_mem_pointer)

    def sample_episode(self, sess, batch_size):
        pointer = np.asarray([np.random.randint(0, self.current_mem_size - 1)])

        print('pointer:', pointer)

        state_ext, state_int, action, reward, state_int_next, episode_length = \
            sess.run(self.get_episode_op,
                     feed_dict={
                         self._mem_pointer_pl: pointer,
                     }
                     )

        return state_ext, state_int, action, reward, state_int_next, episode_length

    def get_episode(self, sess, episode_number):
        """
        Retrieves single episode from memory.
        Returns:
          tuple:
          - dictionary with keys defined by `experience_shape`,
          containing episode records,
          - episode_length scalar.
        """
        assert episode_number < self.current_mem_size

        episode = dict()

        episode['state_external'],
        episode['state_internal'],
        episode['action'],
        episode['reward'],
        episode['state_internal_next'],
        episode_length = sess.run(self.get_episode_op,
                                  feed_dict={self._mem_pointer_pl: episode_number, })

        return episode, episode_length

    def sample_random_experience(self, sess, batch_size=None):
        """
        Samples batch of random experiences from replay memory,
        returns:
            dictionary of O, A, R, O-next, each is np array [batch_size, own_dimension].
        """
        if batch_size is None:
            batch_size =self.batch_size

        try:
            assert batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested memory batch size: {} is bigger than current memory size: {}.'.
                    format(batch_size, self.current_mem_size)
            )

        # Sample episodes:
        episode_indices = np.round(np.random.random_sample(batch_size) * self.current_mem_size).astype(int)
        print('episode_indices:', episode_indices, episode_indices.shape)

        # Get length for each:
        real_length = sess.run(self.get_episodes_length_op,
                               feed_dict={
                                   self.indices1_pl: episode_indices,
                               },
                               )[:, 0]
        #print('episode_length:', real_length, real_length.shape)

        # Sample one experience per episode, from 0 to len -1 to ensure `state_next` exists:
        rnd_multiplier = np.random.random_sample(batch_size)
        experience_indices = np.round((real_length - 1) * rnd_multiplier).astype(int)  # [:,0]
        #print('experience_indices:', experience_indices, experience_indices.shape)

        output_batch = dict(
            state_external=np.zeros((batch_size,) + self.experience_shape['state_external']),
            state_internal=np.zeros((batch_size,) + self.experience_shape['state_internal']),
            action=np.zeros((batch_size,) + self.experience_shape['action']),
            reward=np.zeros((batch_size,) + self.experience_shape['reward']),
            state_external_next=np.zeros((batch_size,) + self.experience_shape['state_external']),
            state_internal_next=np.zeros((batch_size,) + self.experience_shape['state_internal']),
        )

        for i in range(batch_size): # yes, it's a loop and yes, it is slooooow... Any other ideas?
            (
                output_batch['state_external'][i, ...],
                output_batch['state_internal'][i, ...],
                output_batch['action'][i, ...],
                output_batch['reward'][i, ...],
                output_batch['state_internal_next'][i, ...],
            ) = \
                sess.run(
                    self.get_experience_op,
                    feed_dict={
                        self._mem_pointer_pl: episode_indices[i],
                        self._mem_pointer2_pl: experience_indices[i],
                    },
                )

            (output_batch['state_external_next'][i, ...], _1, _2, _3, _4,) = \
                sess.run(
                    self.get_experience_op,
                    feed_dict={
                        self._mem_pointer_pl: episode_indices[i],
                        self._mem_pointer2_pl: experience_indices[i] + 1,
                    }
                )

        return output_batch

    def populate_mempory(self):
        """
        Populates initial replay memory following actual e-greedy policy.
        """

        #while self.current_mem_size < self.replay_memory_init_size:
        #    action = 0
        raise NotImplementedError

    def sample_random_trace(self, sess, batch_size):
        raise NotImplementedError

    def act(self, env, action):
        raise NotImplementedError

    def observe(self, env):
        raise NotImplementedError

    def _build_model_updater(self, estimator1, estimator2, tau):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to update from;
          estimator2: Estimator to be updated.
          tau: update intensity parameter, <<1.
        """
        self.e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        self.e1_params = sorted(self.e1_params, key=lambda v: v.name)
        self.e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        self.e2_params = sorted(self.e2_params, key=lambda v: v.name)

        self.update_model_ops = []
        for e1_p, e2_p in zip(self.e1_params, self.e2_params):
            op = e2_p.assign(e1_p.value() * tau + e2_p.value() * (1 - tau))
            self.update_model_ops.append(op)

    def update_model(self, sess):
        """
        Softly updates model parameters of one estimator towards ones of another.
        sess: Tensorflow session instance
        """
        sess.run(self.update_model_ops)

    def make_epsilon_greedy_policy(self, estimator, nA):
        """
        Creates an epsilon-greedy policy based on a given function approximator and epsilon.

        Args:
            estimator: An estimator that returns q values for a given state
            nA: Number of actions in the environment.

        Returns:
            A function that takes the (sess, observation, epsilon) as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """

        def policy_fn(sess, observation, epsilon):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn
