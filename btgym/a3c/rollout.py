

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.position = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = []
        self.terminal = []
        self.features = []
        self.pixel_change = []
        self.last_actions = []
        self.last_rewards = []

    def add(self,
            position,
            state,
            action,
            reward,
            value,
            value_next,
            terminal,
            features,
            pixel_change,
            last_action,
            last_reward):
        self.position += [position]
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.r += [value_next]
        self.terminal += [terminal]
        self.features += [features]
        self.pixel_change += [pixel_change]
        self.last_actions += [last_action]
        self.last_rewards += [last_reward]

    def add_memory_sample(self, sample):
        """
        Given replay memory sample as list of frames of `length`,
        converts it to rollout of same `length`.
        """
        for frame in sample:
            self.add(
                frame.position,
                frame.state,
                frame.action,
                frame.reward,
                frame.value,
                frame.r,
                frame.terminal,
                frame.features,
                frame.pixel_change,
                frame.last_action,
                frame.last_reward
            )

    """
    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r  # !!
        self.state_next = other.state_next
        self.terminal = other.terminal
        self.features = other.features
        self.pixel_change.extend(other.pixel_change)
        self.last_actions = other.last_action
        self.last_rewards = other.last_reward
    """
