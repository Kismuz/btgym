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


class DictSpace(spaces.Dict):
    """
    Temporal Wrapper for gym Dict space. Adds support for .shape attribute.
    Defines space as nested dictionary of simpler gym spaces.
    """

    def __init__(self, spaces):
        """

        Args:
            spaces_dict:    [nested] dictionary of core Gym spaces.
        """
        super(DictSpace, self).__init__(spaces)
        self.shape = self._get_shape()

    def _get_shape(self):
        return OrderedDict([(k, space.shape) for k, space in self.spaces.items()])


class _DictSpace(Space):
    """
    Defines space as nested dictionary of simpler gym spaces.

    Warning:
        DEPRECATED. Use gym.spaces.Dict instead.
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




