import numpy as np

from btgym.algorithms.utils import batch_stack, batch_gather


class LocalMemory:
    """
    Simple replay buffer allowing random sampling.
    """

    def __init__(self):
        self.on_batch = None
        self.off_batch = None
        self.rp_batch = None

    def reset(self):
        """
        Clears memory.
        """
        self.on_batch = None
        self.off_batch = None
        self.rp_batch = None

    def add_batch(self, on_policy_batch, off_policy_batch, rp_batch):
        """
        Adds data to memory.

        Args:
            data:
        """
        if self.on_batch is None:
            self.on_batch = on_policy_batch

        else:
            self.on_batch = batch_stack([self.on_batch, on_policy_batch])

        if self.off_batch is None:
            self.off_batch = off_policy_batch

        else:
            self.off_batch = batch_stack([self.off_batch, off_policy_batch])

        if self.rp_batch is None:
            self.rp_batch = rp_batch

        else:
            self.rp_batch = batch_stack([self.rp_batch, rp_batch])

    def sample(self, sample_size):
        """
        Randomly samples experiences from memory.

        Args:
            sample_size:

        Returns:
            samples
        """
        if self.on_batch is not None:
            batch_size = self.on_batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            on_policy_batch = batch_gather(self.on_batch, indices)

        else:
            on_policy_batch = None

        if self.off_batch is not None:
            batch_size = self.off_batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            off_policy_batch = batch_gather(self.off_batch, indices)

        else:
            off_policy_batch = None

        if self.rp_batch is not None:
            batch_size = self.rp_batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            rp_batch = batch_gather(self.rp_batch, indices)

        else:
            rp_batch = None

        return {
            'on_policy_batch': on_policy_batch,
            'off_policy_batch': off_policy_batch,
            'rp_batch': rp_batch
        }


class LocalMemory2:
    """
    Simple replay buffer allowing random sampling.
    """

    def __init__(self):
        self.batch = None

    def reset(self):
        """
        Clears memory.
        """
        self.batch = None

    def add_batch(self, on_policy_batch, **kwargs):
        """
        Adds data to memory.

        Args:
            data:
        """
        if self.batch is None:
            self.batch = on_policy_batch

        else:
            self.batch = batch_stack([self.batch, on_policy_batch])

    def sample(self, sample_size):
        """
        Randomly samples experiences from memory.

        Args:
            sample_size:

        Returns:
            samples
        """
        if self.batch is not None:
            batch_size = self.batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            off_policy_batch = batch_gather(self.batch, indices)

        else:
            off_policy_batch = None

        return {
            'on_policy_batch': None,
            'off_policy_batch': off_policy_batch,
            'rp_batch': None
        }

