# This implementation is based on Kosuke Miyoshi code, under Apache License 2.0:
# https://miyosuda.github.io/
# https://github.com/miyosuda/unreal

import numpy as np
from collections import deque

class ExperienceFrame(object):
    def __init__(self,
                 position,
                 state,
                 action,
                 reward,
                 value,
                 r,
                 terminal,
                 features,
                 pixel_change,
                 last_action_reward):
        self.position = position
        self.state = state
        self.action = action  # (Taken action with the 'state')
        self.reward = reward  # Reveived reward with the 'state'.
        self.value = value
        self.r = r
        self.terminal = terminal  # (Whether terminated when 'state' was inputted)
        self.features = features  # LSTM context
        self.pixel_change = pixel_change
        self.last_action_reward = last_action_reward  # (After this last action was taken, agent move to the 'state')


class Memory(object):
    """
    Replay memory.
    """
    def __init__(self, history_size, rp_sequence_size=4, reward_threshold=0.1):
        self._history_size = history_size
        self._frames = deque(maxlen=history_size)
        # Abs. treshold under which frame reward is considered to be zero:
        self.reward_threshold = reward_threshold
        # Reward prediction sampling frame-stack size:
        self.rp_sequence_size = int(rp_sequence_size)
        # Frame indices for zero rewards:
        self._zero_reward_indices = deque()
        # Frame indices for non zero rewards:
        self._non_zero_reward_indices = deque()
        self._top_frame_index = 0

    def add_frame(self, frame):
        """
        Appends single frame to experience buffer.
        """
        if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
            # Discard if terminal frame continues
            print("Sequential terminal frame encountered. Discarded.")
            print(self._frames[-1].position, frame.position)
            return

        frame_index = self._top_frame_index + len(self._frames)
        was_full = self.is_full()

        # Append frame:
        self._frames.append(frame)

        # Decide and append index:
        if frame_index >= self.rp_sequence_size - 1:
            if abs(frame.reward) <= self.reward_threshold:
                self._zero_reward_indices.append(frame_index)

            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            # Decide from which index to discard:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + self.rp_sequence_size - 1
            # Cut frame if its index is lower than cut_frame_index:
            if len(self._zero_reward_indices) > 0 and \
                            self._zero_reward_indices[0] < cut_frame_index:
                self._zero_reward_indices.popleft()

            if len(self._non_zero_reward_indices) > 0 and \
                            self._non_zero_reward_indices[0] < cut_frame_index:
                self._non_zero_reward_indices.popleft()

    def add_rollout(self, rollout):
        """
        Adds frames from given rollout to experience buffer with respect to episode continuation.
        """
        # Check if current rollout is direct extension of last stored frame sequence:
        if len(self._frames) > 0 and not self._frames[-1].terminal:
            # Check if it is same local episode and successive frame order:
            if self._frames[-1].position['episode'] == rollout.position[0]['episode'] and \
                    self._frames[-1].position['step'] + 1 == rollout.position[0]['step']:
                # Means it is ok to just extend previously stored episode
                pass
            else:
                # Means part or tail of previously recorded episode is somehow lost,
                # so need to mark it as 'ended':
                self._frames[-1].terminal = True
                print('***{} changed to terminal'.format(self._frames[-1].position))
                print('*** stored: ', self._frames[-1].position, 'next: ', rollout.position[0])
        # Add experiences one by one:
        # TODO: pain-slow. Vectorize?
        for i in range(len(rollout.position)):
            self.add_frame(
                ExperienceFrame(
                    rollout.position[i],
                    rollout.states[i],
                    rollout.actions[i],
                    rollout.rewards[i],
                    rollout.values[i],
                    rollout.r[i],
                    rollout.terminal[i],
                    rollout.features[i],
                    rollout.pixel_change[i],
                    rollout.last_actions_rewards[i],
                )
            )

    def is_full(self):
        return len(self._frames) >= self._history_size

    def sample_sequence(self, sequence_size):
        """
        Uniformly samples sequence of frames of size `sequence_size`.
        Returns list of frames.
        """
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(0, self._history_size - sequence_size - 1)

        if self._frames[start_pos].terminal:
            start_pos += 1  # assuming that there are no successive terminal frames.

        sampled_frames = []

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_frames.append(frame)
            if frame.terminal:
                break  # it's ok to return less than `sequence_size` frames if `terminal` frame encountered.

        return sampled_frames

    def sample_rp_sequence(self, skewness=2, sample_attempts=100):
        """
        Samples sequence of `self.rp_sequence_size` successive frames for reward prediction,
        prioritizes ones with `non-zero reward' last frame with p=~0.5.
        """
        if np.random.randint(skewness) == 0:
            from_zero = True
        else:
            from_zero = False

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

            start_frame_index = end_frame_index - self.rp_sequence_size + 1
            raw_start_frame_index = start_frame_index - self._top_frame_index

            sampled_frames = []
            is_full = True
            if attempt == sample_attempts - 1:
                check_sequence = False
                print('Warning: failed to sample {} successive frames, sampled as is.'.format(self.rp_sequence_size))

            for i in range(self.rp_sequence_size - 1):
                frame = self._frames[raw_start_frame_index + i]
                sampled_frames.append(frame)
                if check_sequence:
                    if frame.terminal:
                        is_full = False
                        #print('attempt:', attempt)
                        #print('frame.terminal:', frame.terminal)
                        break
            # Last frame can be terminal anyway:
            frame = self._frames[raw_start_frame_index + self.rp_sequence_size - 1]
            sampled_frames.append(frame)

            if is_full:
                break

        return sampled_frames
