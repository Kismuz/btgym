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
            pips_threshold=10,
            pips_scale=1e-4,
            kernel_size=5,
            kernel_stddev=1
    ):
        """

        Args:
            action_space:       actions to advice: 0 - hold, 1- buy, 2 - sell, 3 - close
            time_threshold:     how many points (in number of ENVIRONMENT timesteps) on each side to use
                                for the comparison to consider comparator(n, n+x) to be True
            pips_threshold:     int, minimal peaks difference in pips
                                to consider comparator(n, n+x) to be True
            pips_scale:         actual single pip value wrt signal value
            kernel_size:        gaussian convolution kernel size (used to compute distribution over actions)
            kernel_stddev:      gaussian kernel standart deviation
        """
        self.action_space = action_space
        self.time_threshold = time_threshold
        self.value_threshold = pips_threshold * pips_scale
        self.kernel = signal.gaussian(kernel_size, std=kernel_stddev)
        self.data = None

    def filter_by_margine(self, lst, threshold):
        """
        Filters out peaks by their 'value' difference withing tolerance given.
        Filtering is done from first to last index by removing every succeeding element of list from now on
        if its value difference with value in hand is less than given threshold.

        Args:
            lst:        list of tuples; each tuple is (value, index)
            threshold:  value filtering threshold

        Returns:
            filtered out list of tuples
        """
        if len(lst) == 1:
            return lst
        repeated = abs(lst[1][0] - lst[0][0]) < threshold
        if repeated:
            if len(lst) > 2:
                filtered_tail = self.filter_by_margine([lst[0]] + lst[2:], threshold)
            else:
                filtered_tail = [lst[0]]
        else:
            filtered_tail = [lst[0]] + self.filter_by_margine(lst[1:], threshold)

        return filtered_tail

    def estimate_actions(self, episode_data):
        """
        Estimates hold/buy/sell signals based on data received.

        Args:
            episode_data:   1D np.array of unscaled [but possibly resampled] price values in OHL[CV] format

        Returns:
            1D vector of signals of same length as episode_data
        """
        # Find local maxima and minima indices:
        max_ind = signal.argrelmax(episode_data, order=self.time_threshold)
        min_ind = signal.argrelmin(episode_data, order=self.time_threshold)
        indices = np.append(max_ind, min_ind)
        # Append first and last points:
        indices = np.append(indices, [0, episode_data.shape[0] - 1])
        indices = np.sort(indices)

        indices_and_values = []

        for i in indices:
            indices_and_values.append([episode_data[i], i])

        # Filter by value:
        indices_and_values = self.filter_by_margine(indices_and_values, self.value_threshold)

        #print('filtered_indices_and_values:', indices_and_values)

        # Estimate advised actions (no 'close' btw):
        # Assume all 'hold':
        advice = np.ones(episode_data.shape[0], dtype=np.uint32) * self.action_space[0]

        for num, (v, i) in enumerate(indices_and_values[:-1]):
            if v < indices_and_values[num + 1][0]:
                advice[i] = self.action_space[1]

            else:
                advice[i] = self.action_space[2]

        return advice

    def fit(self, episode_data, resampling_factor=1):
        """
        Estimates actions based on data received.

        Args:
            episode_data:           1D np.array of unscaled price values in OHL[CV] format
            resampling_factor:      factor by which to resample given data
                                    by taking min/max values inside every resampled bar

        Returns:
             Np.array of size [resampled_data_size, actions_space_size] of probabilities of advised actions, where
             resampled_data_size = int(len(episode_data) / resampling_factor) + 1/0

        """
        # Vector of advised actions:
        data = self.resample_data(episode_data, resampling_factor)
        signals = self.estimate_actions(data)

        # One-hot actions encoding:
        actions_one_hot = np.zeros([signals.shape[0], len(self.action_space)])
        actions_one_hot[np.arange(signals.shape[0]), signals] = 1

        # Want a bit relaxed discrete distribution over actions instead of one hot (heuristic):
        actions_distr = np.zeros(actions_one_hot.shape)

        # For all actions except 'hold' (due to heuristic skewness):
        actions_distr[:, 0] = actions_one_hot[:, 0]

        # ...spread out actions probabilities by convolving with gaussian kernel :
        for channel in range(1, actions_one_hot.shape[-1]):
            actions_distr[:, channel] = np.convolve(actions_one_hot[:, channel], self.kernel, mode='same')

        # Normalize:
        actions_distr /= actions_distr.sum(axis=-1)[..., None]

        return actions_distr

    def resample_data(self, episode_data, factor=1):
        """
        Resamples raw observations according to given skip_frame value
        and estimates mean value of newly formed bars.

        Args:
            episode_data:   np.array of shape [episode_length, values]
            factor:     scalar

        Returns:
            np.array of median Hi/Lo observations of size [int(episode_length/skip_frame) + 1, 1]
        """
        # Define resampled size and [possibly] extend
        # to complete last bar by padding with values from very last column:
        resampled_size = int(episode_data.shape[0] / factor)
        #print('episode_data.shape:', episode_data.shape)

        if episode_data.shape[0] / factor > resampled_size:
            resampled_size += 1
            pad_size = resampled_size * factor - episode_data.shape[0]
            #print('pad_size:', pad_size)
            episode_data = np.append(
                episode_data,
                np.zeros([pad_size, episode_data.shape[-1]]) + episode_data[-1, :][None,:],
                axis=0
            )
        #print('new_episode_data.shape:', episode_data.shape)

        # Define HI and Low inside every new bar:
        v_high = np.reshape(episode_data[:,1], [resampled_size, -1]).max(axis=-1)
        v_low = np.reshape(episode_data[:, 2], [resampled_size, -1]).min(axis=-1)

        # ...and yse Hi/Low mean along time_embedding:
        data = np.stack([v_high, v_low], axis=-1).mean(axis=-1)

        return data





