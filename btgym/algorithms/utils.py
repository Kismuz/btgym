
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten as flatten_nested
from tensorflow.python.util.nest import assert_same_structure
from tensorflow.contrib.rnn import LSTMStateTuple

from gym.spaces import Discrete, Dict

from itertools import product


def rnn_placeholders(state):
    """
    Given nested [multilayer] RNN state tensor, infers and returns state placeholders.

    Args:
        state:  tf.nn.lstm zero-state tuple.

    Returns:    tuple of placeholders
    """
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder(tf.float32, tf.TensorShape([None]).concatenate(c.get_shape()[1:]), c.op.name + '_c_pl')
        h = tf.placeholder(tf.float32, tf.TensorShape([None]).concatenate(h.get_shape()[1:]), h.op.name + '_h_pl')
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder(tf.float32, tf.TensorShape([None]).concatenate(h.get_shape()[1:]), h.op.name + '_h_pl')
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)


def nested_placeholders(ob_space, batch_dim=None, name='nested'):
    """
    Given nested observation space as dictionary of shape tuples,
    returns nested state batch-wise placeholders.

    Args:
        ob_space:   [nested] dict of shapes
        name:       name scope
        batch_dim:  batch dimension
    Returns:
        nested dictionary of placeholders
    """
    if isinstance(ob_space, dict):
        out = {key: nested_placeholders(value, batch_dim, name + '_' + key) for key, value in ob_space.items()}
        return out
    else:
        out = tf.placeholder(tf.float32, [batch_dim] + list(ob_space), name + '_pl')
        return out


def nested_discrete_gym_shape(ac_space):
    """
    Given instance of gym.spaces.Dict holding base  gym.spaces.Discrete,
    returns nested dictionary of  spaces depths ( =dict of gym.spaces.Discrete.n)
    This util is here due to fact in practice we need .n attr of discrete space [as cat. encoding depth]
     rather than .shape, which is always ()

    Args:
        ac_space: instance of gym.spaces.Dict

    Returns:
        nested dictionary of lengths
    """
    if isinstance(ac_space, Dict):
        return {key: nested_discrete_gym_shape(space) for key, space in ac_space.spaces.items()}

    elif isinstance(ac_space, Discrete):
        return (ac_space.n,)

    else:
        raise TypeError('Expected gym.spaces.Dict or gym.spaces.Discrete, got: {}'.format(ac_space))


def flat_placeholders(ob_space, batch_dim=None, name='flt'):
    """
    Given nested observation space as dictionary of shape tuples,
    returns flattened dictionary of batch-wise placeholders.

    Args:
        ob_space:   [nested dict] of tuples
        name:       name_scope
        batch_dim:  batch dimension
    Returns:
        flat dictionary of tf.placeholders
    """
    return flatten_nested(nested_placeholders(ob_space, batch_dim=batch_dim, name=name))


def feed_dict_from_nested(placeholder, value, expand_batch=False):
    """
    Zips flat feed dictionary form nested dictionaries of placeholders and values.

    Args:
        placeholder:    nested dictionary of placeholders
        value:          nested dictionary of values
        expand_batch:   if true - add fake batch dimension to values

    Returns:
        flat feed_dict
    """
    assert_same_structure(placeholder, value, check_types=True)
    return _flat_from_nested(placeholder, value, expand_batch)


def _flat_from_nested(placeholder, value, expand_batch):
    feed_dict = {}
    if isinstance(placeholder, dict):
        for key in placeholder.keys():
            feed_dict.update(_flat_from_nested(placeholder[key], value[key], expand_batch))

    else:
        if expand_batch:
            feed_dict.update({placeholder: [value]})

        else:
            feed_dict.update({placeholder: value})

    return feed_dict


def feed_dict_rnn_context(placeholders, values):
    """
    Creates tf.feed_dict for flat placeholders and nested values.

    Args:
        placeholders:       flat structure of placeholders
        values:             nested structure of values

    Returns:
        flat feed dictionary
    """
    return {key: value for key, value in zip(placeholders, flatten_nested(values))}


def as_array(struct):
    """
    Given a dictionary of lists or tuples returns dictionary of np.arrays of same structure.

    Args:
        struct: dictionary of list, tuples etc.

    Returns:
        dict of np.arrays
    """
    if isinstance(struct,dict):
        out = {}
        for key, value in struct.items():
            out[key] = as_array(value)
        return out

    else:
        return np.asarray(struct)


def batch_stack(dict_list, _top=True):
    """
    Stacks values of given processed rollouts along batch dimension.
    Cumulative batch dimension is saved as key 'batch_size' for further shape inference.

    Example:
        dict_list sizes: [[20,10,10,1], [20,10,10,1]] --> result size: [40,10,10,1],
        result['rnn_batch_size'] = 20

    Args:
        dict_list:   list of processed rollouts of the same size.

    Returns:
        dictionary of stacked arrays.
    """
    master = dict_list[0]
    batch = {}

    if isinstance(master, dict):
        for key in master.keys():
            value_list = [value[key] for value in dict_list]
            batch[key] = batch_stack(value_list, False)

    elif isinstance(master, LSTMStateTuple):
        c = batch_stack([state[0] for state in dict_list], False)
        h = batch_stack([state[1] for state in dict_list], False)
        batch = LSTMStateTuple(c=c, h=h)

    elif isinstance(master, tuple):
        batch = tuple([batch_stack([struct[i] for struct in dict_list], False) for i in range(len(master))])

    else:
        try:
            batch = np.concatenate(dict_list, axis=0)

        except ValueError:
            batch = np.stack(dict_list, axis=0)
    if _top:
        # Mind shape inference:
        batch['batch_size'] = batch['batch_size'].sum()
        
    return batch


def batch_gather(batch_dict, indices, _top=True):
    """
    Gathers experiences from processed batch according to specified indices.

    Args:
        batch_dict:     batched data dictionary
        indices:        array-like, indices to gather
        _top:           internal

    Returns:
        batched data of same structure as dict

    """
    batch = {}

    if isinstance(batch_dict, dict):
        for key, value in batch_dict.items():
            batch[key] = batch_gather(value, indices, False)

    elif isinstance(batch_dict, LSTMStateTuple):
        c = batch_gather(batch_dict[0], indices, False)
        h = batch_gather(batch_dict[1], indices, False)
        batch = LSTMStateTuple(c=c, h=h)

    elif isinstance(batch_dict, tuple):
        batch = tuple([batch_gather(struct, indices, False) for struct in batch_dict])

    else:
        batch = np.take(batch_dict, indices=indices, axis=0, mode='wrap')

    if _top:
        # Mind shape inference:
        batch['batch_size'] = indices.shape[0]

    return batch


def batch_pad(batch, to_size, _one_hot=False):
    """
    Pads given `batch` with zeros along zero dimension

    Args:
        batch:      processed rollout as dictionary of np.arrays
        to_size:    desired batch size

    Returns:
        dictionary with all included np.arrays being zero-padded to size [to_size, own_depth].
    """
    if isinstance(batch, dict):
        padded_batch = {}
        for key, struct in batch.items():
            # Mind one-hot action encoding:
            if key in ['action', 'last_action_reward']:
                one_hot = True

            else:
                one_hot = False

            padded_batch[key] = batch_pad(struct, to_size, one_hot)

    elif isinstance(batch, np.ndarray):
        shape = batch.shape
        assert shape[0] < to_size, \
            'Padded batch size must be greater than initial, got: {}, {}'.format(to_size, shape[0])

        pad = np.zeros((to_size - shape[0],) + shape[1:])
        if _one_hot:
            pad[:, 0, ...] = 1
        padded_batch = np.concatenate([batch, pad], axis=0)

    else:
        # Hit tuple, scalar or something else:
        padded_batch = batch

    return padded_batch


def is_subdict(sub_dict, big_dict):
    """
    Checks if first arg is sub_dictionary of second arg by means of structure and values.

    Args:
        sub_dict:       dictionary
        big_dict:       dictionary

    Returns:
        bool
    """
    conditions = []
    if isinstance(sub_dict, dict):
        for key, value in sub_dict.items():
            try:
                conditions.append(is_subdict(value, big_dict[key]))
            except KeyError:
                conditions.append(False)
    else:
        try:
            conditions.append(sub_dict == big_dict)
        except KeyError:
            conditions.append(False)

    return np.asarray(conditions).all()


def _show_struct(struct):
    # Debug utility
    if isinstance(struct, dict):
        for key, value in struct.items():
            print(key)
            _show_struct(value)

    elif type(struct) in [LSTMStateTuple, tuple, list]:
        print('LSTM/tuple/list:', type(struct), len(struct))
        for i in struct:
            _show_struct(i)

    else:
        try:
            print('shape: {}, type: {}'.format(np.asarray(struct).shape, type(struct)))

        except AttributeError:
            print('value:', struct)
