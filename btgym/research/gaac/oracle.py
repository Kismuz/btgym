import numpy as np
from scipy import signal



class Oracle():
    """
    Irresponsible financial adviser.
    """

    def __init__(
            self,
            action_space=(0, 1, 2, 3),
            time_threshold=5,
            value_threshold=0.1,
            kernel_size=5,
            kernel_stddev=1
    ):
        """

        Args:
            action_space:       actions to advice: 0 - hold, 1- buy, 2 - sell, 3 - close
            time_threshold:     how many points on each side to use
                                for the comparison to consider comparator(n, n+x) to be True
            value_threshold:    succeding peaks difference in percentage of normalised signal value
                                to consider comparator(n, n+x) to be True
            kernel_size:        gaussian convolution kernel size (used to compute distribution over actions)
        """
        self.action_space = action_space
        self.time_threshold = time_threshold
        self.value_threshold = value_threshold
        self.kernel = signal.gaussian(kernel_size, std=kernel_stddev)
        self.data = None

    def filter_by_margine(self, lst, tolerance):
        """
        Filters out peaks that lie withing set tolerance

        Args:
            lst:    list of tuples; each tuple is (value, index)
            tolerance:  filterin threshold

        Returns:
            filtered out list of tuples
        """
        if len(lst) == 1:
            return lst
        repeated = abs(lst[1][0] - lst[0][0]) < tolerance
        if repeated:
            if len(lst) > 2:
                filtered_tail = self.filter_by_margine([lst[0]] + lst[2:], tolerance)
            else:
                filtered_tail = [lst[0]]
        else:
            filtered_tail = [lst[0]] + self.filter_by_margine(lst[1:], tolerance)

        return filtered_tail

    def estimate_actions(self, episode_data):
        """

        Args:
            episode_data:   list of episode 'raw_data' observations as OHLC time-embedded data

        Returns:
            vector of advised actions of same length as episode_data
        """
        # Use Hi/Low mean across time_embedding:
        data = [ (np.max(obs[:, 1]) + np.min(obs[:, 2])) / 2 for obs in episode_data]
        data = np.asarray(data)

        # Normalise in  [-1,1]:
        data = 2 * (data - np.max(data)) / - np.ptp(data) - 1

        # Find local maxima and minima indices:
        max_ind = signal.argrelmax(data, order=self.time_threshold)
        min_ind = signal.argrelmin(data, order=self.time_threshold)
        indices = np.append(max_ind, min_ind)
        # Append first and last points:
        indices = np.append(indices, [0, data.shape[0] - 1])
        indices = np.sort(indices)

        indices_and_values = []
        for i in indices:
            indices_and_values.append([data[i], i])

        # Filter by value:
        indices_and_values = self.filter_by_margine(indices_and_values, self.value_threshold)

        # Estimate advised actions (no 'close' btw):
        # Assume all 'hold':
        advice = np.ones(data.shape[0], dtype=np.uint32) * self.action_space[0]

        for num, (v, i) in enumerate(indices_and_values[:-1]):
            if v > indices_and_values[num + 1][0]:
                advice[i] = self.action_space[1]

            else:
                advice[i] = self.action_space[2]

        return advice

    def fit(self, episode_data):
        # Vector of advised actions:
        actions_vec = self.estimate_actions(episode_data)

        # One-hot actions encoding:
        actions_one_hot = np.zeros([actions_vec.shape[0], len(self.action_space)])#, dtype=np.uint32)
        actions_one_hot[np.arange(actions_vec.shape[0]), actions_vec] = 1

        # Make it gaussian distribution over action space:
        actions_distr = np.zeros(actions_one_hot.shape)

        # Convolve for all actions except 'hold' (due to skewnes):
        actions_distr[:, 0] = actions_one_hot[:, 0]
        for channel in range(1, actions_one_hot.shape[-1]):
            actions_distr[:, channel] = np.convolve(actions_one_hot[:, channel], self.kernel, mode='same')

        # Normalize:
        actions_distr /= actions_distr.sum(axis=-1)[..., None]

        return actions_distr






