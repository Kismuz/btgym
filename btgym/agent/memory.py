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
    Sequential/random access replay storage class for Rl agents
    with multi-modal experience and observation state shapes
    and episodic tasks with known maximum length of the episode.
    Stores entire storage as nested dictionary of tf.variables,
    can be saved and restored as part of the model.

    One storage record is `experience` dictionary, defined by `experience_shape`.
    Memory itself consists up to `memory.capacity` number of experiences,
    identified by episodes, with maximum single episode length defined
    by `memory.max_episode_length` value.
    Due to this fact it is possible to extract agent experience in form of [S, A, R, S']
    for every but initial episode step, as well as traces of experiences.

    Experience_shape format:
        can be [nested] dictionary of any structure
        with at least these keys presented at top-level:
            `action`,
            `reward`,
            `done`,
            `state_next`;
        every end-level record is tuple describing tf.variable shape and dtype.
        Shape is arbitrary, dtype can be any of valid numpy-compatible tf.Dtype's. I dtype arg is omitted,
        tf.float32 will be set by default.

    Example:
        robo_experience_shape = dict(
            action=(4,tf.uint8),  # unsigned 8bit integer vector
            reward=(),  # float32 by default, scalar
            done=(tf.bool,),   # boolean, scalar
            state_next=dict(
                internal=dict(
                    hardware=dict(
                        battery=(),  # float32, scalar
                        oil_pressure=(3,),  # got 3 pumps, float32, vector
                        tyre_pressure=(4,),  # for each one, float32, vector
                        checks_passed=(tf.bool,)  # boolean, scalar
                    ),
                    psyche=dict(
                        optimism=(tf.int32,),  # can be high, 32bit int, scalar
                        sincerity=(),  # float32 by default, scalar
                        message=(4,tf.string,),  # knows four phrases
                    )
                ),
                external=dict(
                    camera=(2,180,180,3,tf.uint8),  # binocular rgb 180x180 image, unsigned 8bit integers
                    audio_sensor=(2,320,)  # stereo audio sample buffer, float32
                ),
            ),
            global_training_day=(uint16,)  # just curious how long it took to get to this state.
        )
    """
    capacity = 100000  # in number of experiences
    batch_size = 32  # sampling batch size
    trace_size = 1  # length of continuous experiences trace to sample
    stack_depth = None # stacked observations depth  - alias for sample_stack()
    scope = 'replay_memory'

    def __init__(self,
                 session,
                 experience_shape,  # nested dictionary containing single experience definition.
                 max_episode_length,  # in number of steps, defines maximum buffer capacity
                 **kwargs):
        """______"""
        self.session = session
        self.experience_shape = experience_shape
        self.max_episode_length = max_episode_length

        self.mandatory_keys = ['state_next', 'action', 'reward', 'done']

        # Update defaults:
        for key, value in kwargs.items():
            if key in self.__dir__():
                setattr(self, key, value)

        # Check experience_shape consistency:
        for key in self.mandatory_keys:
            if key not in self.experience_shape:
                msg = (
                    'Mandatory key [{}] not found at top level of `storage.experience_shape` dictionary.\n' +\
                    'Hint: `storage.mandatory_keys` are {}.'
                ).format(key, self.mandatory_keys)
                raise ValueError(msg)

        # Check size consistency:
        try:
            assert self.capacity > self.max_episode_length

        except:
            raise ValueError('Memory capacity <{}> is smaller than maximum single episode length <{}>.'.
                             format(self.capacity, self.max_episode_length))

        try:
            assert self.max_episode_length > self.trace_size

        except:
            raise ValueError('Sample trace size <{}> is bigger than maximum single episode length <{}>.'.
                             format(self.trace_size, self.max_episode_length))

        # Sync:
        if self.stack_depth is not None and self.trace_size != self.stack_depth:
            self.trace_size = self.stack_depth

        else:
            self.stack_depth = self.trace_size

        self.local_step = 0  # step within current episode
        self.local_episode = 0  # keep track of episode numbers within current tf.Session()
        self.current_mem_size = 0  # [points to] stateful tf.variable
        self.current_mem_pointer = -1  # [points to] stateful tf.variable

        # Build logic:
        with tf.variable_scope(self.scope):

            # Define variables and placeholders:
            self._global_variable_constructor()

            # Set storage pointers:
            self._set_mem_cyclic_pointer_op = self._mem_cyclic_pointer.assign(self._mem_cyclic_pointer_pl),
            self._set_mem_current_size_op = self._mem_current_size.assign(self._mem_current_size_pl),

            # Set sync variables:
            self._set_local_step_sync_op = self._local_step_sync.assign(self._local_step_pl)
            self._set_local_episode_sync_op = self._local_episode_sync.assign(self._local_episode_pl)

            # Add single experience to buffer:
            self._add_experience_op = self._feed_buffer_op_constructor(
                self.buffer,
                self.buffer_pl,
                self._local_step_pl,
                scope='add_experience/',
            )
            # Store single episode in storage:
            self._save_episode_op = self._feed_episode_op_constructor(
                self.storage,
                self.buffer,
                self._mem_cyclic_pointer,
                self._position_buffer_pl,
                self._length_pl,
                scope='add_episode/',
            )
            # Get single experience from storage:
            self._get_experience_op = self._get_experience_op_constructor(
                self.storage,
                self._mem_pointer1_pl,
            )
            # Sample batch of random indices:
            self._sample_batch_indices_op = self._sample_indices_batch_op_constructor(
                self.batch_size_pl,
                self.trace_size
            )

            # Get batch of sampled S,A,R,S` experiences traces:
            self._get_sampled_sars_trace_batch_op = self._get_sars_trace_batch_op_constructor(self.indices_batch_pl)

            # Get batch of sampled S,A,R,S` experiences with stacked observations:
            self._get_sars_stack_batch_op = self._get_sars_stack_batch_op_constructor(
                self._get_sampled_sars_trace_batch_op
            )

            # Get batch of singles S,A,R,S` experiences:
            if self.trace_size == 1:
                self._get_sars_random_batch_op = self._get_sars_random_batch_op_constructor(
                    self._get_sampled_sars_trace_batch_op
                )

    def _global_variable_constructor(self):
        """
        Defines TF variables and placeholders.
        """
        with tf.variable_scope('service'):
            # Stateful storage variables:
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
            self._current_episode_id = tf.Variable(
                0,
                trainable=False,
                name='current_id',
                dtype=tf.int32,
            )
            self._local_step_sync = tf.Variable(
                0,
                trainable=False,
                name='local_step_sync',
                dtype=tf.int32,
            )
            self._local_episode_sync = tf.Variable(
                0,
                trainable=False,
                name='local_episode_sync',
                dtype=tf.int32,
            )
            # Indices for retrieving batch of experiences:
            self._batch_indices = tf.Variable(
                tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                trainable=False,
                name='batch_experience_indices',
                dtype=tf.int32,
            )
        # Memory  itself  (as nested dictionary of tensors):
        self.storage = self._var_constructor(
            self.experience_shape,
            self.capacity,
            scope='storage',
        )
        # Add at top-level:
        self.storage['episode_id'] = tf.Variable(
            tf.fill((self.capacity,), value=-1),
            trainable=False,
            name='storage/episode_id',
            dtype=tf.int32,
        )
        # Memory input buffer, accumulates experiences of single episode.
        self.buffer = self._var_constructor(
            self.experience_shape,
            self.max_episode_length,
            scope='buffer',
        )
        # Add at top-level:
        self.buffer['episode_id'] = tf.Variable(
            tf.fill((self.max_episode_length,),value=-1),
            trainable=False,
            name='buffer/episode_id',
            dtype=tf.int32,
        )
        with tf.variable_scope('placeholder'):
            # Placeholders to feed single experience to mem. buffer:
            self.buffer_pl = self._buffer_pl_constructor(
                self.buffer,
                scope='buffer',
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
            self._local_episode_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._indices1_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._indices2_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._position_buffer_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self._length_pl = tf.placeholder(
                dtype=tf.int32,
            )
            self.batch_size_pl = tf.placeholder(
                shape = (),
                dtype=tf.int32,
            )
            self.trace_size_pl = tf.placeholder(
                shape = (),
                dtype=tf.int32,
            )
            self.indices_batch_pl = tf.placeholder(
                shape=(self.batch_size, self.trace_size, 1),
                dtype=tf.int32,
            )

    def _var_constructor(self, shape_dict, memory_capacity, scope='record'):
        """
        Recursive tf.variable constructor.
        Takes:
            shape_dict:
                nested dictionary of tuples in form:
                key_name=(dim_0, dim_1,..., dim_N, [tf.dtype]);
                opt. dtype must be one of tf.DType class object, see:
                https://www.tensorflow.org/api_docs/python/tf/DType
                by default (if no dtype arg. present) is set to: tf.float32;
            memory_capacity:
                maximum possible stored number of experiences, int.scalar ;
            scope:
                top-level name scope.
        Returns:
            nested dictionary of tf.variables of same structure, where every `key` tf.variable has
            name:
                'full_nested_scope/key:0';
            shape:
                (memory_capacity, key_dim_0,..., key_dim_[-1]);
            type:
                consistent tf.Dtype.
        """
        var_dict = dict()
        for key, record in shape_dict.items():
            if type(record) == dict:
                var_dict[key] = self._var_constructor(record, memory_capacity, '{}/{}'.format(scope, str(key)))
            else:
                # If dtype is not present - set it to tf.float32
                dtype = tf.float32
                if len(record) > 0 and type(record[-1]) != int:
                    dtype = record[-1]
                    record = record[0:-1]
                var_dict[key] = tf.Variable(
                    tf.zeros((memory_capacity,) + record, dtype=dtype),
                    trainable=False,
                    name='{}/{}'.format(scope, str(key)),
                    dtype=dtype,
                )
        return var_dict

    def _pl_constructor(self, var_dict, scope='placeholder'):
        """
        Recursive placeholder constructor.
        Takes:
            var_dict:
                nested dictionary of tf.variables;
            scope:
                top-level name scope.
        Returns:
            nested dictionary of placeholders compatible with `var_dict`.
        """
        feed_dict = dict()
        for key, record in var_dict.items():
            if type(record) == dict:
                feed_dict[key] = self._pl_constructor(
                    record,
                    '{}/{}'.format(scope, str(key)),
                )
            else:
                feed_dict[key] = tf.placeholder(
                    dtype=record.dtype,
                    shape=record.shape,
                    name='{}/{}'.format(scope, str(key)),
                )
        return feed_dict

    def _buffer_pl_constructor(self, buffer_dict, scope='buffer_pl'):
        """
        Defines placeholders to feed single experience to storage buffer.
        Takes:
            buffer_dict:
                nested dictionary of tf.variables;
           scope:
                top-level name scope.
        Returns:
            nested dictionary of placeholders compatible with `buffer_dict`, with
            rank of each placeholder reduced by one in expense of removing (episode_length) dimension.
        """
        feed_dict = dict()
        for key, record in buffer_dict.items():
            if type(record) == dict:
                feed_dict[key] = self._buffer_pl_constructor(
                    record,
                    '{}/{}'.format(scope, str(key)),
                )
            else:
                feed_dict[key] = tf.placeholder(
                    dtype=record.dtype,
                    shape=record.shape[1:],
                    name='{}/{}'.format(scope, str(key)),
                )
        return feed_dict

    def _feed_buffer_op_constructor(self, var_dict, pl_dict, step_pl, scope='add_experience'):
        """
        Defines operations to store single experience in storage buffer.
        Takes:
            var_dict:
                nested dictionary of tf.variables;
            pl_dict:
                nested dictionary of placeholders of consisted structure.
            step_pl:
                local step of the episode, scalar;
            scope:
                top-level name scope.
        Returns:
            nested dictionary of `tf.assign` operations.
        """
        op_dict = dict()
        for key, var in var_dict.items():
            if type(var) == dict:
                op_dict[key] = self._feed_buffer_op_constructor(
                    var,
                    pl_dict[key],
                    step_pl,
                    '{}/{}'.format(scope, str(key)),
                )
            else:
                op_dict[key] = tf.assign(
                    var[step_pl, ...],
                    pl_dict[key],
                    name='{}/{}'.format(scope, str(key)),
                )
        return op_dict

    def _feed_episode_op_constructor(
            self,
            memory_dict,
            buffer_dict,
            position_memory_pl,
            position_buffer_pl,
            length_pl,
            scope='add_episode',
    ):
        """
        Defines operations to store [part of] single episode to storage.
        Takes:
            memory_dict:
                nested dictionary of tf.variables;
            pl_dict:
                nested dictionary of placeholders of same as `var_dict` structure;
            position_memory_pl:
                start position in storage to write episode to, int64 scalar;
            position_buffer_pl:
                start position in buffer to write from, int64 scalar;
            length_pl:
                [partial] episode length_pl, int64 scalar.
            scope:
                top-level name scope.
        Returns:
            nested dictionary of operations.
        """
        op_dict = dict()
        for key, var in memory_dict.items():
            if type(var) == dict:
                op_dict[key] = self._feed_episode_op_constructor(
                    var,
                    buffer_dict[key],
                    position_memory_pl,
                    position_buffer_pl,
                    length_pl,
                    '{}/{}'.format(scope, str(key)),
                )
            else:
                op_dict[key] = tf.assign(
                    var[position_memory_pl: position_memory_pl + length_pl, ...],
                    buffer_dict[key][position_buffer_pl: position_buffer_pl + length_pl, ...],
                    name='{}/{}'.format(scope, str(key)),
                )
        return op_dict

    def _get_experience_op_constructor(self, memory_dict, position1_pl):
        """
        Defines ops to retrieve single experience from storage.
        Takes:
            memory_dict:
                nested dictionary of tf.variables;
            position1_pl:
                place in storage to get experience from, scalar;
        Returns:
            nested dictionary of sliced tensors.
        """
        get_dict = dict()
        for key, var in memory_dict.items():
            if type(var) == dict:
                get_dict[key] = self._get_experience_op_constructor(
                    var,
                    position1_pl,
                )
            else:
                get_dict[key] = memory_dict[key][position1_pl, ...]
        return get_dict

    def _get_experience_batch_op_constructor(self, memory_dict, batch_indices):
        """
        Defines operations to retrieve batch of experiences from storage.
        Takes:
            memory_dict:
                [nested] dictionary of tf.variables;
            batch_indices:
                tensor of indices of shape (batch_size, trace_length, 1)
        Returns:
            nested dictionary of sliced tensors of shape: (batch_size, trace_length, key_field_own_dimensions).
        """
        batch_dict = dict()
        for key, var in memory_dict.items():
            if type(var) == dict:
                batch_dict[key] = self._get_experience_batch_op_constructor(
                    var,
                    batch_indices,
                )
            else:
                batch_dict[key] = tf.gather_nd(
                    memory_dict[key],
                    batch_indices,
                )
        return batch_dict

    def _sample_indices_batch_op_constructor(self, batch_size, trace_size):
        """
        Defines operations for sampling random batch of experiences's indices.
        # Args:
            batch_size: int64 scalar.
            trace_size: int64 scalar.
        # Returns:
            tensor of indices in shape (output_batch_size, trace_size, 1),
            where output_batch_size <= batch_size
        Note:
            - initial experiences are excluded from sampling range;
            - trace is defined as continuous sequence of experiences of same episode;
            - this method op only samples once and rejects inconsistent traces, one should
            run it [possibly] several times to add up to batch of needed size.
        """
        # Sample global indices:
        rnd_idx = tf.random_uniform(
            shape=(batch_size,),
            minval=trace_size + 1,  # +2  is here to exclude initial observations
            maxval=self._mem_current_size,
            dtype=tf.int32,
        )
        # Make trace-back matrix:
        idx_trace = tf.transpose(
            tf.multiply(
                tf.ones(
                    (batch_size, trace_size + 1),
                    dtype=tf.int32
                ),
                tf.cast(
                    tf.range(0, trace_size + 1),
                    dtype=tf.int32,
                )
            )
        )
        # Expand sampled indices:
        rnd_idx_expanded = tf.multiply(
            tf.ones(
                (trace_size + 1, batch_size),
                dtype=tf.int32
            ),
            rnd_idx,
        ),  # <-- !!! it makes tuple, and it works. WTF?

        # Trace-back sampled indices:
        rnd_idx_traced = tf.subtract(
            rnd_idx_expanded,
            idx_trace,
        )
        # Accept continuous traces only, e.g. from same episode:
        accept_trace_condition = tf.equal(
            tf.gather_nd(
                self.storage['episode_id'],
                tf.transpose(rnd_idx_traced)
            ),
            tf.gather_nd(
                self.storage['episode_id'],
                tf.transpose(rnd_idx_expanded)
            )
        )
        accept_trace_all = tf.reduce_all(
            accept_trace_condition,
            axis=1
        )
        # Get accepted samples indices in-batch:
        accepted_idx_in_batch = tf.boolean_mask(
            tf.cast(
                tf.range(0, batch_size),
                dtype=tf.int32,
            ),
            accept_trace_all
        )
        # Map it to storage indices:
        accepted_global_indices = tf.gather(
            tf.transpose(rnd_idx_traced),
            accepted_idx_in_batch,
        )
        return tf.reverse(accepted_global_indices[:, :-1, :], axis=[1])  # trim zero-values and reverse

    def _get_sars_trace_batch_op_constructor(self, batch_indices):
        """
        Defines operations for getting batch of experiences in S,A,R,S` form.
        Returns:
            dictionary of operations.
        """
        # Get `-1` indices for `state` field:
        previous_mask = tf.stack(
            tf.ones(shape=batch_indices.shape[1:], dtype=tf.int32)
        )
        batch_indices_previous = tf.subtract(
            batch_indices,
            previous_mask,
        )
        # Get >-A,R,S` part:
        sars_batch = self._get_experience_batch_op_constructor(
            self.storage,
            batch_indices,
        )
        # Get S,-> part:
        sars_batch['state'] = self._get_experience_batch_op_constructor(
            {'fake_key': self.storage['state_next']},
            batch_indices_previous,
        )['fake_key']
        return sars_batch

    def _get_sars_stack_batch_op_constructor(self, input_batch, stack=False):
        """
        Defines operations for converting batch of traces in S,A,R,S` to random batch with
        'state' and 'state_next' records stacked along last dimension (atari-style).
        # Args:
            input_batch:  batch of S,A,R,S` traces.
        # Returns:
            dictionary of operations.
        """
        op_dict = dict()
        for key, record in input_batch.items():
            if type(record) == dict:
                if key in ['state', 'state_next']:
                    stacked = True

                else:
                    stacked = stack

                op_dict[key] = self._get_sars_stack_batch_op_constructor(record, stacked)

            else:
                if stack:
                    op_dict[key] = tf.stack(
                        [record[:, i, ...] for i in range(self.trace_size)],
                        axis=-1
                    )
                else:
                    op_dict[key] = record[:, -1, ...]

        return op_dict

    def _get_sars_random_batch_op_constructor(self, input_batch):
        """
        Defines operations for removing redundant [second] dimension when trace_size = 1.
        # Returns:
            dictionary of operations.
        """
        op_dict = dict()
        for key, record in input_batch.items():
            if type(record) == dict:
                op_dict[key] = self._get_sars_random_batch_op_constructor(record)

            else:
                op_dict[key] = tf.squeeze(
                    record,
                    axis=1
                )
        return op_dict

    def _make_feeder(self, pl_dict, value_dict):
        """
        Makes `serialized` feed dictionary.
        Takes:
            pl_dict:
                nested dictionary of tf.placeholders;
            value_dict:
                dictionary of values of same as `pl_dict` structure.
        Returns:
            flattened feed dictionary, tf.Session.run()-ready.
        """
        feeder = dict()
        for key, record in pl_dict.items():
            if type(record) == dict:
                feeder.update(self._make_feeder(record, value_dict[key]))
            else:
                feeder.update({record: value_dict[key]})
        return feeder

    def evaluate_buffer(self, buffer_dict):
        """
        Handy if something goes wrong.
        """
        content_dict = dict()
        for key, var in buffer_dict.items():
            if type(var) == dict:
                content_dict[key] = self.evaluate_buffer(var)
            else:
                content_dict[key] = self.session.run(var)
        return content_dict

    def print_nested_dict(self, nested_dict, tab=''):
        """
        Handy.
        """
        for k, v in nested_dict.items():
            if type(v) == dict:
                print('{}{}:'.format(tab, k))
                self.print_nested_dict(v, tab + '   ')
            else:
                print('{}{}:'.format(tab, k))
                print('{}{}'.format(tab + tab, v))

    def print_global_variables(self):
        """
        Handy.
        """
        for v in tf.global_variables():
            print(v)

    def _get_current_size(self):
        """
        Returns current used storage size
        in number of stored episodes, in range: [0, max_mem_size).
        """
        return self.session.run(self._mem_current_size)

    def _set_current_size(self, value):
        """
        Sets current used storage size in number of records (experiences).
        """
        assert value <= self.capacity
        _ = self.session.run(
            self._set_mem_current_size_op,
            feed_dict={
                self._mem_current_size_pl: value,
            }
        )

    def _get_cyclic_pointer(self):
        """
        Cyclic pointer stores number (==address) of episode in replay storage,
        currently to be written/replaced.
        This pointer supposed to infinitely loop through entire storage, updating records.
        Returns:
            current pointer value.
        """
        return self.session.run(self._mem_cyclic_pointer)

    def _set_cyclic_pointer(self, value):
        """
        Sets one.
        """
        _ = self.session.run(self._set_mem_cyclic_pointer_op,
                     feed_dict={self._mem_cyclic_pointer_pl: value},
                     )

    def _get_local_step_sync(self):
        """
        Returns current counter value.
        """
        return self.session.run(self._local_step_sync)

    def _set_local_step_sync(self, value):
        """
        Sets one.
        """
        _ = self.session.run(
            self._set_local_step_sync_op,
            feed_dict={
                self._local_step_pl: value
            }
        )

    def _get_local_episode_sync(self):
        """
        Returns current counter value.
        """
        return self.session.run(self._local_episode_sync)

    def _set_local_episode_sync(self, value):
        """
        Sets one.
        """
        _ = self.session.run(
            self._set_local_episode_sync_op,
            feed_dict={
                self._local_episode_pl: value
            }
        )

    def _add_experience(self, experience):
        """
        Writes single experience to episode storage buffer and
        calls add_episode() method if experience['done']=True
        or maximum storage episode length exceeded.
        Receives:
            experience: dictionary containing agent experience,
                        shaped according to self.experience_shape.
        """
        # Check local and stateful steps for consistency. If not in sync -> tf model
        # possibly been reloaded.
        if self.local_step != self._get_local_step_sync()\
            or self.local_episode != self._get_local_episode_sync():
            # Huston... we've lost somewhere. Better rewind:
            print('Memory buffer unsync found:\nLocal: step_{}, episode_{}.\nModel: step_{}, episode_{}.\nCorrected.'.
                  format(self.local_step,
                         self.local_episode,
                         self._get_local_step_sync(),
                         self._get_local_episode_sync()
                         )
                  )
            self.local_step = 0
            self._set_local_step_sync(self.local_step)
            self.local_episode = self._get_local_episode_sync()

        # Get 'done' flag:
        done = experience['done']

        # Add experience id:
        experience.update({'episode_id': self.local_episode})

        # If we haven't run out of buffer capacity:
        if self.local_step < self.max_episode_length:
            # Prepare feeder dict:
            feeder = self._make_feeder(
                pl_dict=self.buffer_pl,
                value_dict=experience,
            )
            # Add local step:
            feeder.update({self._local_step_pl: self.local_step})
            # TODO: Where is episode id?

            # Save it:
            _ = self.session.run(
                self._add_experience_op,
                feed_dict=feeder,
            )
        else:
            done = True
            self.local_step -= 1
        if done:
            # If over, store episode in replay storage:
            self._add_episode()
        else:
            self.local_step += 1
            self._set_local_step_sync(self.local_step)

    def _add_episode(self):
        """
        Writes episode to replay storage.
        """
        # Get actual storage size and pointer:
        self.current_mem_size = self._get_current_size()
        self.current_mem_pointer = self._get_cyclic_pointer()
        # Save:
        # If we need to split episode or not:
        if self.current_mem_pointer + self.local_step < self.capacity:
            print('-->', self.current_mem_pointer, self.local_step)
            # Do not split, write entire episode:
            _ = self.session.run(
                self._save_episode_op,
                feed_dict={
                    self._position_buffer_pl: 0,
                    self._length_pl: self.local_step + 1,
                }
            )
            # Update pointers:
            if self.current_mem_size < self.capacity:
                self.current_mem_size += self.local_step  + 1 #???????
                self._set_current_size(self.current_mem_size)

            self.current_mem_pointer += self.local_step + 1
            self._set_cyclic_pointer(self.current_mem_pointer)

        else:
            print('SPLIT-->', self.current_mem_pointer, self.local_step)
            # Split episode, write tail to storage beginning:
            body = self.capacity - self.current_mem_pointer
            tail = self.local_step - body
            _1 = self.session.run(
                self._save_episode_op,
                feed_dict={
                    self._position_buffer_pl: 0,
                    self._length_pl: body  #+ 1,
                }
            )
            # Rewind:
            self.current_mem_pointer = 0
            self._set_cyclic_pointer(self.current_mem_pointer)

            _2 = self.session.run(
                self._save_episode_op,
                feed_dict={
                    self._position_buffer_pl: body,
                    self._length_pl: tail + 1,
                }
            )
            # Update pointers:
            self.current_mem_pointer = tail + 1
            self._set_cyclic_pointer(self.current_mem_pointer)

            # Obviously, reached full capacity:
            self.current_mem_size = self.capacity
            self._set_current_size(self.current_mem_size)

        # Reset local_step, increment episode count:
        self.local_step = 0
        self._set_local_step_sync(self.local_step)
        self.local_episode += 1
        if self.local_episode == 2147483647:
            self.local_episode = 0
        self._set_local_episode_sync(self.local_episode)

    def get_episode(self, episode_number):
        """
        Retrieves single episode from storage.
        Returns:
            dictionary with keys defined by `experience_shape`,
            containing episode records.
        """
        try:
            assert episode_number <= self._get_current_size()

        except:
            raise ValueError('Episode index <{}> is out of storage bounds <{}>.'.
                             format(episode_number, self._get_current_size()))
        raise NotImplementedError

    def update(self, experience):
        """
        Wrapper method for adding single experience to storage. Is here fo future edit.
        Note: it's essential to pass correct experience[`done`] value
        in order to ensure correct episode storage in storage,
        e.g. when forcefully terminating episode before `done` is sent by environment.
        """
        self._add_experience(experience)

    def _sample_batch_indices(self, batch_size, trace_size, depth=0):
        """
        Samples indices of experience traces.
        # Args:
            batch_size: int32, scalar.
            trace_size: int32, scalar.
        # Returns:
            batch_indices: int32 np.array holding sampled storage indices,
                           shaped (batch_size, trace_size, 1).
        """
        # Sanity check:
        if depth > 100:
            raise  RecursionError('Can not acquire sample of batch size {} after {} attempts'.
                                  format(batch_size, depth))

        indices_batch = self.session.run(
            self._sample_batch_indices_op,
            feed_dict={
                self.batch_size_pl: batch_size,
                #self.trace_size_pl: trace_size
            }
        )
        batch_shortage = batch_size - indices_batch.shape[0]
        if batch_shortage > 0:
            #print('batch_shortage:', batch_shortage)
            indices_batch_2 = self._sample_batch_indices(batch_shortage, trace_size, depth+1)
            #print('batch2: ',indices_batch_2.shape)
            indices_batch = np.concatenate((indices_batch, indices_batch_2), axis=0)

        return indices_batch

    def sample_trace(self):
        """
        Samples batch of random experiences traces from replay storage.
            - method is stateful: every call will return new sample.
            - initial experiences are excluded from sampling range.
            - trace is defined as continuous sequence of (S,A,R,S`)'es of same episode.
        # Returns:
            nested dictionary, holding batches of traces of corresponding storage experience fields:
            (S, A, R, S-next), each one is np.array of shape (output_batch_size, trace_size, field_own_dimension).
        # Note:
            - when self.trace_size = 1 [default] --> sample is  batch of random experiences
              with second output dimension == 1.
            - `temporal` dimension of the output is second one (keras.TimeDistributed-style).
        """
        # TODO: can return dict of tensors itself, for direct connection with estimator input.
        self.current_mem_size = self._get_current_size()
        try:
            assert self.batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested storage batch of size {} can not be sampled: storage contains {} records.'.
                format(self.batch_size, self.current_mem_size)
            )
        # Sample & retrieve:
        output_dict = self.session.run(
            self._get_sampled_sars_trace_batch_op,
            feed_dict={
                self.indices_batch_pl: self._sample_batch_indices(self.batch_size, self.trace_size)
            }
        )
        return output_dict

    def sample_stack(self):
        """
        Samples batch of single experiences with stacked observations (atari-style time embedding):
        # Returns:
            nested dictionary, holding batches of random (S, A, R, S-next) experiences
            with 'state' and 'state_next' observations stacked along last dimension:
            (output_batch_size, field_own_dimension, stack_depth).
        # Note:
            - S_stacked = (S_-n, S_-n+1, ..., S_0), n = stack_depth.
            - 'temporal' dimension of the output is last one (image_channel-style).
        """
        try:
            assert self.batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested storage batch of size {} can not be sampled: storage contains {} records.'.
                format(self.batch_size, self.current_mem_size)
            )
        batch_dict = self.session.run(
            self._get_sars_stack_batch_op,
            feed_dict={
                self.indices_batch_pl: self._sample_batch_indices(self.batch_size, self.trace_size)
            }
        )
        return batch_dict

    def sample_random(self):
        """
        Samples batch of single (S, A, R, S-next) experiences from memory.
        # Returns:
            nested dictionary, holding batches of random (S, A, R, S-next) experiences:
            for every record field dimension is (output_batch_size, field_own_dimension).
        # Note:
             - assert memory.trace_size or memory.stack_depth are set to 1.
             - this method is equivalent to sample_trace() with redundant second dimension removed from output.
         """
        try:
            assert self.trace_size == 1

        except:
            raise AssertionError('Method `sample_random()` requires trace_size OR stack_depth == 1, got: {}'.
                                 format(self.trace_size))
        try:
            assert self.batch_size <= self.current_mem_size

        except:
            raise AssertionError(
                'Requested storage batch of size {} can not be sampled: storage contains {} records.'.
                format(self.batch_size, self.current_mem_size)
            )
        batch_dict = self.session.run(
            self._get_sars_random_batch_op,
            feed_dict={
                self.indices_batch_pl: self._sample_batch_indices(self.batch_size, self.trace_size)
            }
        )
        return batch_dict
