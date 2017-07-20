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

import numpy as np
import tensorflow as tf


class BTgymReplayMemory():
    """
    Sequential/random access replay memory class for experiences
    with multi-modal state observation.
    Stores entire memory as dictionary of tf.variables, each tensor
    defines one memory field;
    can be saved and restored as part of tensorflow model.

    Current version supports bi-modal observations.
    Bi-modal observation state shape is defined
    as dictionary of two arbitrary shaped tensors:
    state_shape = dict(external=(N1,N2, ..., Nk),
                       internal=(M1, M2, ..., Ml),)

    Shape of single experience therefore defined by memory fields(=keys):
    experience_shape = dict(state_external=state_shape['external'],
                            state_internal=state_shape['internal'],
                            action=(),
                            reward=(),
                            state_internal_next=state_shape['internal'],
                            state_external_next=state_shape['internal'],)
    """

    def __init__(self,
                 state_shape,  # dictionary of external and internal shapes
                 max_episode_length,  # in number of steps
                 max_size=100000,  # in number of experiences
                 batch_size=32,
                 scope='replay_memory',):
        """______"""
        self.state_shape = state_shape
        self.experience_shape = dict(
            state_external=state_shape['external'],
            state_internal=state_shape['internal'],
            action=(),
            reward=(),
            done=(),
            state_external_next=state_shape['external'],
            state_internal_next=state_shape['internal'],
        )
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.memory_shape = (int(max_size / max_episode_length), self.max_episode_length)

        try:
            assert self.memory_shape[0] > 0

        except:
            raise ValueError('Memory maximum size <{}> is smaller than maximum single episode length <{}>.:'.
                                 format(max_size, self.max_episode_length))

        self.local_step = 0  # step within current episode
        self.episode = 0  # keep track of eisode numbers within current tf.Session()
        self.current_mem_size = 0  # stateful
        self.current_mem_pointer = -1  # stateful

        # Build graphs:
        with tf.variable_scope(scope):
            self._tf_variable_constructor()
            self._tf_graph_constructor()

    def _tf_variable_constructor(self):
        """
        Defines TF variables and placeholders.
        """
        with tf.variable_scope('placeholder'):

            # Placeholders to feed single experience to mem. buffer:
            self.buff_feeder = dict(
                state_external=tf.placeholder(
                    dtype=tf.float32,
                    shape=self.experience_shape['state_external'],
                ),
                state_internal=tf.placeholder(
                    dtype=tf.float32,
                    shape=self.experience_shape['state_internal'],
                ),
                action=tf.placeholder(
                    dtype=tf.int32,
                    shape=self.experience_shape['action'],
                ),
                reward=tf.placeholder(
                    dtype=tf.float32,
                    shape=self.experience_shape['reward'],
                ),
                state_internal_next=tf.placeholder(
                    dtype=tf.float32,
                    shape=self.experience_shape['state_internal_next'],
                ),
                episode_length=tf.placeholder(
                    dtype=tf.int32,
                    shape=(1,),
                ),
            )
            # Placeholders for service variables:
            self._mem_current_size_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._mem_cyclic_pointer_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._mem_pointer1_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._mem_pointer2_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._local_step_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._indices1_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._indices2_pl = tf.placeholder(
                dtype=tf.int32,
            )

            with tf.variable_scope('buffer'):
                # Memory buffer accumulates experiences of single episode.
                self.buffer = dict(
                    state_external=tf.Variable(
                        tf.zeros((self.max_episode_length,) + self.experience_shape['state_external']),
                        trainable=False,
                        name='state_external',
                        dtype=tf.float32,
                    ),
                    state_internal=tf.Variable(
                        tf.zeros((self.max_episode_length,) + self.experience_shape['state_internal']),
                        trainable=False,
                        name='state_internal',
                        dtype=tf.float32,
                    ),
                    action=tf.Variable(
                        tf.zeros((self.max_episode_length,) + self.experience_shape['action'], dtype=tf.int32),
                        trainable=False,
                        name='action',
                        dtype=tf.int32,
                    ),
                    reward=tf.Variable(
                        tf.zeros((self.max_episode_length,) + self.experience_shape['reward']),
                        trainable=False,
                        name='reward',
                        dtype=tf.float32,
                    ),
                    state_internal_next=tf.Variable(
                        tf.zeros((self.max_episode_length,) + self.experience_shape['state_internal_next']),
                        trainable=False,
                        name='state_internal_next',
                        dtype=tf.float32,
                    ),
                    episode_length=tf.Variable(
                        tf.zeros((1,), dtype=tf.int32),
                        trainable=False,
                        name='episode_length',
                        dtype=tf.int32,
                    ),
                )

            with tf.variable_scope('memory_variable'):
                # Stateful memory variables:
                self._mem_current_size = tf.Variable(  # in number of episodes
                    0,
                    trainable=False,
                    name='current_size',
                    dtype=tf.int32,
                )
                self._mem_cyclic_pointer = tf.Variable(
                    0,
                    trainable=False,
                    name='cyclic_pointer',
                    dtype=tf.int32,
                )
                # Indices for retrieving batch of experiences:
                self._batch_indices = tf.Variable(
                    tf.zeros(shape=(self.batch_size, 2), dtype=tf.int32),
                    trainable=False,
                    name='batch_experience_indices',
                    dtype=tf.int32,
                )
                # Memory  itself  (as dictionary of tensors):
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

    def _tf_graph_constructor(self):
        """
        Defines TF graphs and handles.
        """
        # Set memory pointers:
        self._set_mem_cyclic_pointer_op = self._mem_cyclic_pointer.assign(self._mem_cyclic_pointer_pl),
        self._set_mem_current_size_op = self._mem_current_size.assign(self._mem_current_size_pl),

        # Add single experience to buffer:
        self._add_experience_op = [self.buffer['episode_length'].assign(self.buff_feeder['episode_length'])]
        for key, var in self.buffer.items():
            if key != 'episode_length':
                op = var[self._local_step_pl, ...].assign(self.buff_feeder[key])
                self._add_experience_op.append(op)

        # Store single episode in memory:
        self._save_episode_op = []
        for key, var in self.memory.items():
            op = var[self._mem_cyclic_pointer, ...].assign(self.buffer[key])
            self._save_episode_op.append(op)

        # Get single episode from memory:
        self._get_episode_op = []
        episode_length = self.memory['episode_length'][self._mem_pointer1_pl]
        for key, var in self.memory.items():
            op = var[self._mem_pointer1_pl, 0: episode_length[0], ...]
            self._get_episode_op.append(op)

        # Gather episode's length values by its numbers:
        self._get_episodes_length_op = tf.gather(self.memory['episode_length'], self._indices1_pl)

        # Get single experience from memory:
        self._get_experience_op = []
        for key, var in self.memory.items():
            if key != 'episode_length':
                op = var[self._mem_pointer1_pl, self._mem_pointer2_pl, ...]
                self._get_experience_op.append(op)

        # Get batch of random experiences:
        self._sample_batch_indices_op = []

        # Sample episode numbers:
        self._sample_batch_indices_op.append(
            self._batch_indices[:, 0].assign(
                tf.random_uniform(
                    shape=(self.batch_size,),
                    minval=0,
                    maxval=self._mem_current_size,
                    dtype=tf.int32,
                )
            )
        )
        # Get real length value for each episode, reduce by one to ensure next_state exists:
        episode_len_values = tf.gather(
            self.memory['episode_length'],
            self._batch_indices[:, 0],
        )[:, 0] - 1

        # Now can sample experiences indices:
        self._sample_batch_indices_op.append(
            self._batch_indices[:, 1].assign(
                tf.cast(
                    tf.multiply(
                        tf.cast(
                            episode_len_values,
                            dtype=tf.float32,
                        ),
                        tf.random_uniform(
                            shape=(self.batch_size,),
                            #minval=0,
                            #maxval=1,
                            dtype=tf.float32,
                        )
                    ),
                    dtype=tf.int32,
                )
            )
        )
        # Get indices for 'state_external_next' field:
        next_mask = tf.stack(
            [
                tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                tf.ones(shape=(self.batch_size,), dtype=tf.int32),
            ],
            axis=1,
        )
        batch_indices_next = tf.add(
            self._batch_indices,
            next_mask
        )
        # Dictionary of tensors containing batch of experiences for every memory field:
        self._get_experience_batch_op = dict(
            state_external=None,
            state_internal=None,
            action=None,
            reward=None,
            state_external_next=None,
            state_internal_next=None,
        )
        # Define 'memory read' operations:
        for key, memory in self.memory.items():
            if key != 'episode_length':
                self._get_experience_batch_op[key] = tf.gather_nd(
                    self.memory[key],
                    self._batch_indices,
                )

        self._get_experience_batch_op['state_external_next'] = tf.gather_nd(
            self.memory['state_external'],
            batch_indices_next,
        )

    def _evaluate_buffer(self, sess):
        """
        Handy if something goes wrong.
        """
        content = []
        for key, var in self.buffer.items():
            content.append(sess.run(var))
        return  content

    def _get_memory_size(self, sess):
        """
        Returns current used memory size
        in number of stored episodes, in range [0, max_mem_size].
        """
        # TODO: use .eval(sess)?
        return sess.run(self._mem_current_size)

    def _set_memory_size(self, sess, value):
        """
        Sets current used memory size in number of episodes.
        """
        assert value <= self.memory_shape[0]
        _ = sess.run(self._set_mem_current_size_op,
                     feed_dict={
                         self._mem_current_size_pl: value,
                     }
                     )

    def _get_cyclic_pointer(self, sess):
        """
        Cyclic pointer stores number (==address) of episode in replay memory,
        currently to be written/replaced.
        This pointer supposed to infinitely loop through entire memory, updating records.
        Returns: current pointer value.
        """
        return sess.run(self._mem_cyclic_pointer)

    def _set_cyclic_pointer(self, sess, value):
        """
        Sets one.
        """
        _ = sess.run(self._set_mem_cyclic_pointer_op,
                     feed_dict={self._mem_cyclic_pointer_pl: value},
                     )

    def add_experience(self, sess, experience):
        """
        Writes single experience to episode memory buffer,
        calls add_episode() method if experience['done']=True.
        Receives:
            sess:       tf.Session object,
            experience: dictionary containing agent experience,
                        shaped according to self.experience_shape.
        """
        # Prepare feeder dict:
        feeder = {self._local_step_pl: self.local_step}
        experience['episode_length'] = np.asarray([self.local_step]).astype(int)
        for key, value in self.buffer.items():
            feeder.update({self.buff_feeder[key]: experience[key]})

        #for k,v in feeder.items():
        #    print('feeder: {}: {}\n'.format(k,v))

        # Save
        sess.run(
            self._add_experience_op,
            feed_dict=feeder,
        )
        self.local_step += 1

        if experience['done'] or self.local_step >= self.memory_shape[1]:
            # If over, write episode to replay memory:
            self.add_episode(sess)

    def add_episode(self, sess):
        """
        Writes episode to replay memory.
        """
        # Save:
        sess.run(self._save_episode_op, )

        # print('Saved episode with:')
        # print('mem_size:', self.get_memory_size(sess))
        # print('cyclic_pointer:', self.get_cyclic_pointer(sess))
        # Reset local_step, increment episode count:
        self.local_step = 0
        self.episode += 1

        # Increment memory size and move cycle_pointer to next episode:
        # Get actual size and pointer:
        self.current_mem_size = self._get_memory_size(sess)
        self.current_mem_pointer = self._get_cyclic_pointer(sess)

        if self.current_mem_size < self.memory_shape[0] - 1:
            # If memory is not full - increase used size by 1,
            # else - leave it along:
            self.current_mem_size += 1
            self._set_memory_size(sess, self.current_mem_size)

        if self.current_mem_pointer >= self.current_mem_size:
            # Rewind cyclic pointer, if reached memory upper bound:
            self._set_cyclic_pointer(sess, 0)
            self.current_mem_pointer = 0

        else:
            # Increment:
            self.current_mem_pointer += 1
            self._set_cyclic_pointer(sess, self.current_mem_pointer)

    def get_episode(self, sess, episode_number):
        """
        Retrieves single episode from memory.
        Returns:
          tuple:
          - dictionary with keys defined by `experience_shape`,
            containing episode records,
          - episode_length scalar.
        """
        try:
            assert episode_number <= self._get_memory_size(sess)

        except:
            raise ValueError('Episode index <{}> is out of memory bounds <{}>.'.
                             format(episode_number, self._get_memory_size(sess)))

        episode = dict()
        (episode['state_external'],
        episode['state_internal'],
        episode['action'],
        episode['reward'],
        episode['state_internal_next'],
        episode_length) =\
            sess.run(
                self._get_episode_op,
                feed_dict={self._mem_pointer1_pl: episode_number, })

        return episode, episode_length

    def _sample_random_experience(self, sess, batch_size=None):
        """
        DEPRECATED.
        Samples batch of random experiences from replay memory,
        returns:
            dictionary of O, A, R, O-next, each is np array [batch_size, own_dimension].
        """
        self.current_mem_size = self._get_memory_size(sess)
        if batch_size is None:
            batch_size = self.batch_size

        try:
            assert batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested memory batch of size {} can not be sampled from current memory size: {} episodes.'.
                format(batch_size, self.current_mem_size)
            )
        # Sample episodes:
        episode_indices = (np.random.random_sample(batch_size) * self.current_mem_size).astype(int)

        ###print('episode_indices:', episode_indices, episode_indices.shape)

        # Get length for each:
        real_length = sess.run(self._get_episodes_length_op,
                               feed_dict={
                                   self._indices1_pl: episode_indices,
                               },
                               )[:, 0]

        ###print('episode_length:', real_length, real_length.shape)

        # Sample one experience per episode, from 0 to len -1 to ensure `state_next` exists:
        rnd_multiplier = np.random.random_sample(batch_size)
        experience_indices = (real_length  * rnd_multiplier).astype(int)  # [:,0]

        ###print('experience_indices:', experience_indices, experience_indices.shape)

        output_batch = dict(
            state_external=np.zeros((batch_size,) + self.experience_shape['state_external']),
            state_internal=np.zeros((batch_size,) + self.experience_shape['state_internal']),
            action=np.zeros((batch_size,) + self.experience_shape['action']),
            reward=np.zeros((batch_size,) + self.experience_shape['reward']),
            state_external_next=np.zeros((batch_size,) + self.experience_shape['state_external']),
            state_internal_next=np.zeros((batch_size,) + self.experience_shape['state_internal']),
        )

        for i in range(batch_size):  # yes, it's a loop and yes, it is slooooow... Any other ideas?
            (
                output_batch['state_external'][i, ...],
                output_batch['state_internal'][i, ...],
                output_batch['action'][i, ...],
                output_batch['reward'][i, ...],
                output_batch['state_internal_next'][i, ...],
            ) = \
                sess.run(
                    self._get_experience_op,
                    feed_dict={
                        self._mem_pointer1_pl: episode_indices[i],
                        self._mem_pointer2_pl: experience_indices[i],
                    },
                )

            (output_batch['state_external_next'][i, ...], _1, _2, _3, _4,) = \
                sess.run(
                    self._get_experience_op,
                    feed_dict={
                        self._mem_pointer1_pl: episode_indices[i],
                        self._mem_pointer2_pl: experience_indices[i] + 1,
                    }
                )
        return output_batch

    def sample_random_trace(self, sess, batch_size):
        raise NotImplementedError

    def sample_random_experience(self, sess, as_tensors=False):
        """
        Samples batch of random experiences from replay memory,
        returns:
            dictionary, every key holds  batch of corresponding memory field experiences:
            O, A, R, O-next, each one is np.array of shape [batch_size, field_own_dimension].
        """
        # TODO: can return dict of tensors itself, for direct connection with estimator input
        self.current_mem_size = self._get_memory_size(sess)

        try:
            assert self.batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested memory batch of size {} can not be sampled: memory contains {} episodes.'.
                format(self.batch_size, self.current_mem_size)
            )
        # Sample indices:
        _ = sess.run(self._sample_batch_indices_op)

        # Retrieve values :
        output_feeder = dict()
        for key, tensor in self._get_experience_batch_op.items():
            output_feeder.update(
                {key: sess.run(tensor)}
            )

        return output_feeder

