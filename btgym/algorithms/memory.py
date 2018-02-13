# This implementation is based on Kosuke Miyoshi code, under Apache License 2.0:
# https://miyosuda.github.io/
# https://github.com/miyosuda/unreal

from logbook import Logger, StreamHandler, WARNING
import sys

import numpy as np
from collections import deque
from btgym.algorithms.rollout import Rollout


class Memory(object):
    """
    Replay memory with rebalanced replay based on reward value.

    Note:
        must be filled up before calling sampling methods.
    """
    def __init__(self, history_size, max_sample_size, priority_sample_size, log_level=WARNING,
                 rollout_provider=None, task=-1, reward_threshold=0.1, use_priority_sampling=False):
        """

        Args:
            history_size:           number of experiences stored;
            max_sample_size:        maximum allowed sample size (e.g. off-policy rollout length);
            priority_sample_size:   sample size of priority_sample() method
            log_level:              int, logbook.level;
            rollout_provider:       callable returning list of Rollouts NOT USED
            task:                   parent worker id;
            reward_threshold:       if |experience.reward| > reward_threshold: experience is saved as 'prioritized';
        """
        self._history_size = history_size
        self._frames = deque(maxlen=history_size)
        self.reward_threshold = reward_threshold
        self.max_sample_size = int(max_sample_size)
        self.priority_sample_size = int(priority_sample_size)
        self.rollout_provider = rollout_provider
        self.task = task
        self.log_level = log_level
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('ReplayMemory_{}'.format(self.task), level=self.log_level)
        self.use_priority_sampling = use_priority_sampling
        # Indices for non-priority frames:
        self._zero_reward_indices = deque()
        # Indices for priority frames:
        self._non_zero_reward_indices = deque()
        self._top_frame_index = 0

        if use_priority_sampling:
            self.sample_priority = self._sample_priority

        else:
            self.sample_priority = self._sample_dummy

    def add(self, frame):
        """
        Appends single experience frame to memory.

        Args:
            frame:  dictionary of values.
        """
        if frame['terminal'] and len(self._frames) > 0 and self._frames[-1]['terminal']:
            # Discard if terminal frame continues
            self.log.warning("Memory_{}: Sequential terminal frame encountered. Discarded.".format(self.task))
            self.log.warning('{} -- {}'.format(self._frames[-1]['position'], frame['position']))
            return

        frame_index = self._top_frame_index + len(self._frames)
        was_full = self.is_full()

        # Append frame:
        self._frames.append(frame)

        # Decide and append index:
        if frame_index >= self.max_sample_size - 1:
            if abs(frame['reward']) <= self.reward_threshold:
                self._zero_reward_indices.append(frame_index)

            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            # Decide from which index to discard:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + self.max_sample_size - 1
            # Cut frame if its index is lower than cut_frame_index:
            if len(self._zero_reward_indices) > 0 and \
                            self._zero_reward_indices[0] < cut_frame_index:
                self._zero_reward_indices.popleft()

            if len(self._non_zero_reward_indices) > 0 and \
                            self._non_zero_reward_indices[0] < cut_frame_index:
                self._non_zero_reward_indices.popleft()

    def add_rollout(self, rollout):
        """
        Adds frames from given rollout to memory with respect to episode continuation.

        Args:
            rollout:    `Rollout` instance.
        """
        # Check if current rollout is direct extension of last stored frame sequence:
        if len(self._frames) > 0 and not self._frames[-1]['terminal']:
            # E.g. check if it is same local episode and successive frame order:
            if self._frames[-1]['position']['episode'] == rollout['position']['episode'][0] and \
                    self._frames[-1]['position']['step'] + 1 == rollout['position']['step'][0]:
                # Means it is ok to just extend previously stored episode
                pass
            else:
                # Means part or tail of previously recorded episode is somehow lost,
                # so we need to mark stored episode as 'ended':
                self._frames[-1]['terminal'] = True
                self.log.warning('{} changed to terminal'.format(self._frames[-1]['position']))
                # If we get a lot of such messages it is an indication something is going wrong.
        # Add experiences one by one:
        # TODO: pain-slow.
        for i in range(len(rollout['terminal'])):
            frame = rollout.get_frame(i)
            self.add(frame)

    def is_full(self):
        return len(self._frames) >= self._history_size

    def fill(self):
        """
        Fills replay memory with initial experiences. NOT USED.
        Supposed to be called by parent worker() just before training begins.

        Args:
            rollout_getter:     callable, returning list of Rollouts.

        """
        if self.rollout_provider is not None:
            while not self.is_full():
                for rollout in self.rollout_provider():
                    self.add_rollout(rollout)
            self.log.info('Memory_{}: filled.'.format(self.task))

        else:
            raise AttributeError('Rollout_provider is None, can not fill memory.')

    def sample_uniform(self, sequence_size):
        """
        Uniformly samples sequence of successive frames of size `sequence_size` or less (~off-policy rollout).

        Args:
            sequence_size:  maximum sample size.
        Returns:
            instance of Rollout of size <= sequence_size.
        """
        start_pos = np.random.randint(0, self._history_size - sequence_size - 1)
        # Shift by one if hit terminal frame:
        if self._frames[start_pos]['terminal']:
            start_pos += 1  # assuming that there are no successive terminal frames.

        sampled_rollout = Rollout()

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_rollout.add(frame)
            if frame['terminal']:
                break  # it's ok to return less than `sequence_size` frames if `terminal` frame encountered.

        return sampled_rollout

    def _sample_priority(self, size=None, exact_size=False, skewness=2, sample_attempts=100):
        """
        Implements rebalanced replay.
        Samples sequence of successive frames from distribution skewed by means of reward of last sample frame.

        Args:
            size:               sample size, must be <= self.max_sample_size;
            exact_size:         whether accept sample with size less than 'size'
                                or re-sample to get sample of exact size (used for reward prediction task);
            skewness:           int>=1, sampling probability denominator, such as probability of sampling sequence with
                                last frame having non-zero reward is: p[non_zero]=1/skewness;
            sample_attempts:    if exact_size=True, sets number of re-sampling attempts
                                to get sample of continuous experiences (no `Terminal` frames inside except last one);
                                if number is reached - sample returned 'as is'.
        Returns:
            instance of Rollout().
        """
        if size is None:
            size = self.priority_sample_size

        if size > self.max_sample_size:
            size = self.max_sample_size

        # Toss skewed coin:
        if np.random.randint(int(skewness)) == 0:
            from_zero = False
        else:
            from_zero = True

        if len(self._zero_reward_indices) == 0:
            # zero rewards container was empty
            from_zero = False
        elif len(self._non_zero_reward_indices) == 0:
            # non zero rewards container was empty
            from_zero = True

        # Try to sample sequence of given length from one episode.
        # Take maximum of 'sample_attempts', if no luck
        # (e.g too short episodes and/or too big sampling size) ->
        # return inconsistent sample and issue warning.
        check_sequence = True
        for attempt in range(sample_attempts):
            if from_zero:
                index = np.random.randint(len(self._zero_reward_indices))
                end_frame_index = self._zero_reward_indices[index]

            else:
                index = np.random.randint(len(self._non_zero_reward_indices))
                end_frame_index = self._non_zero_reward_indices[index]

            start_frame_index = end_frame_index - size + 1
            raw_start_frame_index = start_frame_index - self._top_frame_index

            sampled_rollout = Rollout()
            is_full = True
            if attempt == sample_attempts - 1:
                check_sequence = False
                self.log.warning(
                    'Memory_{}: failed to sample {} successive frames, sampled as is.'.format(self.task, size)
                )

            for i in range(size - 1):
                frame = self._frames[raw_start_frame_index + i]
                sampled_rollout.add(frame)
                if check_sequence:
                    if frame['terminal']:
                        if exact_size:
                            is_full = False
                        #print('attempt:', attempt)
                        #print('frame.terminal:', frame['terminal'])
                        break
            # Last frame can be terminal anyway:
            frame = self._frames[raw_start_frame_index + size - 1]
            sampled_rollout.add(frame)

            if is_full:
                break

        return sampled_rollout

    @staticmethod
    def _sample_dummy(**kwargs):
        return None


class _DummyMemory:

    def __init__(self):
        pass

    @staticmethod
    def add(frame):
        return None

    @staticmethod
    def sample_uniform(**kwargs):
        return None

    @staticmethod
    def sample_priority(**kwargs):
        return  None

    @staticmethod
    def is_full():
        return True