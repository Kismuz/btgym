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

class BTgymMultiSpace(Space):
    """
    Multi-modal observation space wrapper.
    Defines space as flat [not nested] dictionary of spaces.
    """

    def __init__(self, spaces_dict):
        """

        Args:
            spaces_dict:    dictionary of core Gym spaces.
        """
        self.spaces = spaces_dict

    def sample(self):
        """
        Uniformly randomly sample a random element of this space.

        Returns:
            dictionary of samples
        """
        return {key: space.sample() for key, space in self.spaces.items()}

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        try:
            for key, space in x.items():
                if not self.spaces[key].contains(space):
                    return False
            return True

        except:
            return False

    def get_shapes(self):
        """
        Get shapes for every space included.

        Returns:
            dictionary of shapes

        """
        return {key: self.spaces[key].shape for key in self.spaces.keys()}

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n