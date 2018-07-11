###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin, muzikinae@gmail.com
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

from gym import Space
from gym import spaces

from collections import OrderedDict
from itertools import product

from numpy import asarray


class DictSpace(spaces.Dict):
    """
    Wrapper for gym.spaces.Dict class. Adds support for .shape attribute.
    Defines space as nested dictionary of simpler gym spaces.
    """

    def __init__(self, spaces_dict):
        """

        Args:
            spaces_dict:    [nested] dictionary of core Gym spaces.
        """
        super(DictSpace, self).__init__(spaces_dict)
        self.shape = self._get_shape()

    def _get_shape(self):
        return OrderedDict([(k, space.shape) for k, space in self.spaces.items()])


class ActionDictSpace(DictSpace):
    """
    Action space for btgym environments as shallow dictionary of discrete or continuous spaces.
    Defines several handy attributes and encoding conversion methods.
    """

    def __init__(self, assets, base_actions=None,):
        """

        Args:
            base_actions:   None or iterable of base asset discrete actions;
                            if no actions provided - continuous 1D base action space is set in [0,1] interval.
            assets:         iterable of assets names
        """
        self.assets = tuple(sorted(assets))
        if base_actions is not None:
            self.base_actions = tuple(base_actions)
            self.base_actions_lookup_table = dict(list(enumerate(self.base_actions)))
            self.base_space = spaces.Discrete
            self.tensor_shape = (len(self.assets), len(self.base_actions))
            self.lookup_table = self.make_lookup_table(
                base_actions=list(self.base_actions_lookup_table.keys()),
                num_assets=len(self.assets)
            )
            self.cardinality = len(list(self.lookup_table.keys()))
            self.depth = self.cardinality
            spaces_dict = {key: spaces.Discrete(self.tensor_shape[-1]) for key in self.assets}

        else:
            self.base_actions = None
            self.base_actions_lookup_table = None
            self.base_space = spaces.Box
            self.tensor_shape = (len(self.assets), 1)
            self.lookup_table = None
            self.cardinality = None
            self.depth = self.tensor_shape[0]
            spaces_dict = {
                key: spaces.Box(low=0, high=1, shape=(self.tensor_shape[-1],), dtype='float32') for key in self.assets
            }
        super(ActionDictSpace, self).__init__(spaces_dict)

    @staticmethod
    def make_lookup_table(base_actions, num_assets):
        """
        Creates lookup table for set of environment actions for K assets
        and N base actions as a cartesian product of K sets of N elements each.

        Args:
            base_actions:   iterable of base asset actions
            num_assets:     int, number of assets

        Returns:
            lookup table as dictionary form {num_0: env_action_0, ...}
        """
        return dict(list(enumerate(product(list(base_actions), repeat=num_assets))))

    def action_to_vec(self, action):
        """
        Given action returns its vector encoding.

        Args:
            action:     action from this space (shallow dictionary)

        Returns:
            numpy array
        """
        assert self.contains(action), 'Action {} does not belongs to this space'.format(action)
        return asarray([action[key] for key in self.assets])

    def vec_to_action(self, vector):
        """
        Given vector encoding of an action returns action from this space.

        Args:
            vector:     iterable of scalars

        Returns:
            action as shallow dictionary of scalars
        """
        assert len(vector) == len(self.assets), \
            'Length of encoding and number of assets should match, got: {} / {}'.format(len(vector), len(self.assets))

        action = OrderedDict([(asset, value) for asset, value in zip(self.assets, vector)])

        assert self.contains(action), 'Vector {} can not be converted to action of this space'.format(vector)
        return action

    def vec_to_cat(self, action):
        """
        Given action vector returns it's position (categorical encoding).
        Only for dictionary of discrete base spaces.

        Args:
            action:     environment action as tuple, list or array of base asset cations

        Returns:
            int, position in lookup table

        Raises:
            ValueError, if no matches found
        """
        assert self.lookup_table is not None, 'Lookup table not defined for base {}'.format(self.base_space)

        for key, value in self.lookup_table.items():
            if list(value) == list(action):
                return key
        raise ValueError('Action vector {} is not in lookup table of this space.'.format(action))

    def cat_to_vec(self, category):
        """
        Given integer as categorical encoding returns corresponding env. action vector.
        Only for dictionary of discrete base spaces.

        Args:
            category:   int, encoding
            table:      lookup table

        Returns:
            environment action as numpy array of base asset actions

        Raises:
            ValueError, if no matches found

        """
        assert self.lookup_table is not None, 'Lookup table not defined for base {}'.format(self.base_space)
        try:
            return asarray(self.lookup_table[category])

        except KeyError:
            raise ValueError('Category {} does not match action space.'.format(category))


class _DictSpace(Space):
    """
    Defines space as nested dictionary of simpler gym spaces.

    """
    def __init__(self, spaces_dict):
        """

        Args:
            spaces_dict:    [nested] dictionary of core Gym spaces.
        """
        self._nested_map(self._make_assert_gym_space(), spaces_dict)
        self.spaces = spaces_dict
        self.shape = self._nested_shape()

    @staticmethod
    def _gym_spaces():
        attr_names = [attr for attr in dir(spaces) if attr[0].isupper()]
        return tuple([getattr(spaces, name) for name in attr_names])

    @staticmethod
    def _contains(space, sample):
        return space.contains(sample)

    @staticmethod
    def _shape(space, *args):
        return space.shape

    @staticmethod
    def _sample(space, *args):
        return space.sample()

    def _make_assert_gym_space(self):
        gym_spaces = self._gym_spaces()

        def assert_gym_space(space, *args):
            try:
                assert isinstance(space, gym_spaces)

            except:
                raise AssertionError('Space {} is not valid Gym space'.format(type(space)))

        return assert_gym_space

    def _nested_contains(self, x):
        try:
            self._assert_structure(self.spaces, x)
            return self._nested_map(self._contains, self.spaces, x)

        except:
            return False

    def _nested_shape(self):
        return self._nested_map(self._shape, self.spaces)

    def _nested_sample(self):
        return self._nested_map(self._sample, self.spaces)

    def _assert_structure(self, s1, s2):
        if isinstance(s1, dict) or isinstance(s2, dict):
            try:
                assert isinstance(s1, dict) and isinstance(s2, dict)

            except:
                raise AssertionError('Args are not of the same structure. Got arg1: {}, arg2: {}'.
                                     format(type(s1), type(s2)))
            keys1 = set(s1.keys())
            keys2 = set(s2.keys())
            for key in keys1 | keys2:
                try:
                    assert key in keys1

                except:
                    raise AssertionError('Key <{}> not present in arg1'.format(key))

                try:
                    assert key in keys2

                except:
                    raise AssertionError('Key <{}> not present in arg2'.format(key))

                self._assert_structure(s1[key], s2[key])

    def _nested_map(self, func, struct, *arg):
        if not callable(func):
            raise TypeError('`func` arg. must be callable.')

        if len(arg) == 0:
            struct2 = struct

        else:
            struct2 = arg[0]

        if isinstance(struct, dict):
            mapped = {key: self._nested_map(func, struct[key], struct2[key]) for key in struct.keys()}

        else:
            mapped = func(struct, struct2)

        return mapped

    def sample(self):
        """
        Uniformly randomly sample a random element of this space.

        Returns:
            dictionary of samples
        """
        return self._nested_sample()

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return self._nested_contains(x)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        #return sample_n
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        #return sample_n
        raise NotImplementedError




